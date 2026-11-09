"""Conservative unison restoration from accompaniment motifs and stereo audio.

This does not recover two independently identifiable recordings. It tests
whether an existing lead attack also fits an established accompaniment line
and its measured same-pitch spatial component. References are never inputs.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .audio import load_audio_channels
from .tab import _candidates, playable


def _tone_spectrum(data, sr, note):
    start = max(0, round((note['start']+.01)*sr))
    end = min(len(data), round(min(note['end'], note['start']+.18)*sr))
    chunk = data[start:end]
    if len(chunk) < 32:
        return np.zeros((0, 2), complex)
    spectrum = np.fft.rfft(chunk*np.hanning(len(chunk))[:, None], n=16384, axis=0)
    frequency = 440*2**((note['pitch']-69)/12)
    bins = np.fft.rfftfreq(16384, 1/sr)
    return spectrum[abs(bins-frequency) < max(4., frequency*.012)]


def _spatial_vector(spectra):
    matrices = []
    for spectrum in spectra:
        covariance = spectrum.T @ spectrum.conj()
        energy = float(np.trace(covariance).real)
        if energy > 1e-12:
            matrices.append(covariance/energy)
    if len(matrices) < 2:
        return None
    _, vectors = np.linalg.eigh(np.mean(matrices, axis=0))
    return vectors[:, -1]


def recover_unisons(path, lead, rhythm, raw_notes, features, opts):
    """Return rhythm, restored notes and auditable proof for each hypothesis.

    All motifs and spatial models use the original frozen notes. Restored
    notes never become evidence for another restoration. Existing notes keep
    their pitch, position, duration and fingering.
    """
    if len(raw_notes) != len(features):
        raise ValueError('Unison features must align with raw detections')
    if not lead or not rhythm:
        return list(rhythm), [], []
    grid = opts.quantize*60/max(opts.bpm, 1)
    if grid <= 0:
        raise ValueError('Quantization must be positive')
    step = lambda n: round((n['start']-opts.grid_offset)/grid)
    by_key = defaultdict(list)
    for i, n in enumerate(raw_notes):
        by_key[n['pitch'], n['confidence']].append(i)
    def index(note):
        found = by_key[note['pitch'], note['confidence']]
        return found[0] if len(found) == 1 else None
    def feature(note):
        i = index(note)
        return features[i] if i is not None else None
    def valid(note):
        f = feature(note)
        return f is not None and f.valid and f.spatial_rank >= .1
    lead_cells = {(step(n), n['pitch']) for n in lead}
    rhythm_cells = {(step(n), n['pitch']) for n in rhythm}
    hypotheses = []
    for note in lead:
        if (step(note), note['pitch']) in rhythm_cells or not valid(note):
            continue
        local_lead = [n for n in lead if valid(n) and abs(n['start']-note['start']) <= 8.]
        local_rhythm = [n for n in rhythm if valid(n) and abs(n['start']-note['start']) <= 8.]
        if len(local_lead) < 3:
            continue
        lead_pan = float(np.median([feature(n).pan for n in local_lead]))
        opposed = [n for n in local_rhythm if abs(feature(n).pan-lead_pan) >= .65]
        if len(opposed) < 3:
            continue
        backing_pan = float(np.median([feature(n).pan for n in opposed]))
        if abs(lead_pan-backing_pan) < .65:
            continue
        backing = [n for n in local_rhythm if abs(feature(n).pan-backing_pan)
                   <= abs(feature(n).pan-lead_pan)+.15]
        cells = defaultdict(list)
        for n in backing:
            cells[step(n)].append(n)
        patterns = defaultdict(set)
        for n in backing:
            cell, pitch = step(n), n['pitch']
            if (cell, pitch) in lead_cells:
                continue
            for gap in range(1, 5):
                for a in cells[cell+gap]:
                    for b in cells[cell+2*gap]:
                        if len({pitch % 12, a['pitch'] % 12, b['pitch'] % 12}) == 3:
                            patterns[pitch, a['pitch'], b['pitch'], gap].add(cell)
        proofs = []
        cell = step(note)
        for gap in range(1, 5):
            for a in cells[cell+gap]:
                for b in cells[cell+2*gap]:
                    anchors = patterns.get((note['pitch'], a['pitch'], b['pitch'], gap), set())
                    if len(anchors) >= 3:
                        proofs.append(dict(gap_steps=gap, following_pitches=[a['pitch'], b['pitch']],
                                           anchor_cells=sorted(anchors)))
        if proofs:
            hypotheses.append(dict(note=note, raw_index=index(note), restored=False, proofs=proofs))
    if not hypotheses:
        return list(rhythm), [], []
    data, sr = load_audio_channels(path)
    if data.shape[1] != 2:
        return list(rhythm), [], []
    if not np.isfinite(data).all():
        raise ValueError('Audio contains non-finite samples')
    hypothesis_cells = {(step(h['note']), h['note']['pitch']) for h in hypotheses}
    spectra = {}
    def spectrum(note):
        i = index(note)
        if i not in spectra:
            spectra[i] = _tone_spectrum(data, sr, raw_notes[i])
        return spectra[i]
    restored = []
    for h in hypotheses:
        note = h['note']
        anchors = set().union(*(set(p['anchor_cells']) for p in h['proofs']))
        lead_refs = [n for n in lead if n['pitch'] == note['pitch'] and valid(n)
                     and abs(n['start']-note['start']) <= 8.
                     and (step(n), n['pitch']) not in hypothesis_cells
                     and (step(n), n['pitch']) not in rhythm_cells]
        rhythm_refs = [n for n in rhythm if n['pitch'] == note['pitch'] and valid(n)
                       and step(n) in anchors and (step(n), n['pitch']) not in lead_cells]
        left, right = _spatial_vector([spectrum(n) for n in lead_refs]), _spatial_vector([spectrum(n) for n in rhythm_refs])
        h.update(lead_reference_count=len(lead_refs), rhythm_reference_count=len(rhythm_refs))
        if left is None or right is None:
            h['reason'] = 'insufficient_same_pitch_audio'
            continue
        matrix = np.column_stack([left, right])
        condition = float(np.linalg.cond(matrix))
        h['spatial_condition'] = condition
        if not np.isfinite(condition) or condition > 4.:
            h['reason'] = 'indistinguishable_spatial_models'
            continue
        # Check the attack at its final TAB cell. The detector can report a
        # later start, after a quieter unison component has already decayed.
        projected = _tone_spectrum(data, sr, note) @ np.linalg.inv(matrix).T
        energy = np.sum(abs(projected)**2, axis=0)
        fractions = energy/max(float(energy.sum()), 1e-15)
        h['component_fractions'] = [float(x) for x in fractions]
        if energy.sum() <= 1e-12 or min(fractions) < .05:
            h['reason'] = 'only_one_source_supported'
            continue
        cell = step(note)
        sounding = [n for n in rhythm+restored if n['start'] <= note['start']+1e-6 < n['end']]
        shape = [(n['string'], n['fret']) for n in sounding]
        positions = [p for p in _candidates(note['pitch'], opts) if playable(tuple(shape+[p]), opts)]
        if not positions:
            h['reason'] = 'no_playable_position'
            continue
        string, fret = min(positions, key=lambda p:(p[1] != 0, abs(p[1]-note['fret']), p))
        next_start = min((n['start'] for n in rhythm if n['start'] > note['start']+1e-6), default=note['end'])
        added = dict(note, string=string, fret=fret, end=min(note['end'], next_start))
        restored.append(added)
        h.update(restored=True, reason='motif_and_two_spatial_components', restored_note=added)
    return sorted([dict(n) for n in rhythm]+restored, key=lambda n:n['start']), restored, hypotheses
