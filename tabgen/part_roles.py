"""Recheck ambiguous guitar roles using motifs and frequency-specific stereo.

The models compare an existing lead fundamental with the same frequency in
an accompaniment harmonic. They are local evidence, not isolated recordings
or calibrated instrument probabilities. No reference TAB is an input.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .audio import load_audio_channels
from .tab import arrange_melody
from .unison import _spatial_vector, _tone_spectrum


def refine_part_roles(path, lead, rhythm, raw_notes, features, opts):
    if len(raw_notes) != len(features):
        raise ValueError('Role features must align with raw detections')
    if not lead or not rhythm:
        return list(lead), list(rhythm), []
    grid = opts.quantize*60/max(opts.bpm, 1)
    if grid <= 0:
        raise ValueError('Quantization must be positive')
    step = lambda n: round((n['start']-opts.grid_offset)/grid)
    by_key = defaultdict(list)
    for i, n in enumerate(raw_notes):
        by_key[n['pitch'], n['confidence']].append(i)
    def index(n):
        found = by_key[n['pitch'], n['confidence']]
        return found[0] if len(found) == 1 else None
    def valid(n):
        i = index(n)
        return i is not None and features[i].valid and features[i].spatial_rank >= .1
    cells = {step(n): n for n in lead}
    rhythm_cells = {(step(n), n['pitch']) for n in rhythm}
    patterns = defaultdict(set)
    for cell, note in cells.items():
        for gap in range(1, 5):
            a, b = cells.get(cell-gap), cells.get(cell+gap)
            if (a and b and valid(note) and valid(a) and valid(b)
                    and len({a['pitch'] % 12, note['pitch'] % 12, b['pitch'] % 12}) == 3):
                patterns[a['pitch'], note['pitch'], b['pitch'], gap].add(cell)
    hypotheses = []
    for i, n in enumerate(raw_notes):
        cell = step(n)
        if (not valid(n) or n['confidence'] < opts.min_confidence
                or n['start'] < opts.lead_start_seconds
                or (cell in cells and cells[cell]['pitch'] == n['pitch'])):
            continue
        proofs = []
        for gap in range(1, 5):
            a, b = cells.get(cell-gap), cells.get(cell+gap)
            if not a or not b:
                continue
            anchors = patterns.get((a['pitch'], n['pitch'], b['pitch'], gap), set())
            local = [s for s in anchors if abs(s-cell)*grid <= 8.]
            if len(local) < 3:
                continue
            current = cells.get(cell)
            if current:
                existing = patterns.get((a['pitch'], current['pitch'], b['pitch'], gap), set())
                if len([s for s in existing if abs(s-cell)*grid <= 8.]) >= 3:
                    continue
                # Protect unique melody pitches. A replacement must leave the
                # displaced pitch audible in the already established backing.
                if (cell, current['pitch']) not in rhythm_cells:
                    continue
            proofs.append(dict(gap_steps=gap, previous_pitch=a['pitch'], next_pitch=b['pitch'],
                               anchor_cells=sorted(local)))
        if proofs:
            hypotheses.append(dict(raw_index=i, cell=cell, proofs=proofs, accepted=False))
    if not hypotheses:
        return list(lead), list(rhythm), []
    data, sr = load_audio_channels(path)
    if data.shape[1] != 2:
        return list(lead), list(rhythm), []
    if not np.isfinite(data).all():
        raise ValueError('Audio contains non-finite samples')
    candidate_cells = {h['cell'] for h in hypotheses}
    spectra = {}
    def spectrum(n, pitch):
        i = index(n)
        key = (i, pitch)
        if key not in spectra:
            spectra[key] = _tone_spectrum(data, sr, dict(raw_notes[i], pitch=pitch))
        return spectra[key]
    choices = {}
    for h in hypotheses:
        note = raw_notes[h['raw_index']]
        cell = h['cell']
        local_lead = [n for n in lead if valid(n) and abs(n['start']-note['start']) <= 8.]
        if len(local_lead) < 3:
            h['reason'] = 'insufficient_voice_context'
            continue
        lead_pan = float(np.median([features[index(n)].pan for n in local_lead]))
        carriers = sorted({n['pitch'] for n in rhythm if step(n) == cell and any(
            abs(1200*np.log2(harmonic)-100*(note['pitch']-n['pitch'])) < 25
            for harmonic in (2, 3, 4))})
        models = []
        for carrier in carriers:
            backing_refs = [n for n in rhythm if n['pitch'] == carrier and valid(n)
                and abs(n['start']-note['start']) <= 8. and step(n) not in candidate_cells
                and abs(features[index(n)].pan-lead_pan) >= .5]
            backing_refs = [n for n in backing_refs if not any(
                l['pitch'] == note['pitch'] and abs(l['start']-n['start']) <= .08 for l in lead)]
            lead_refs = [n for n in local_lead if n['pitch'] == note['pitch']
                and step(n) not in candidate_cells and not any(
                    b['pitch'] == carrier and valid(b)
                    and abs(features[index(b)].pan-lead_pan) >= .5
                    and abs(b['start']-n['start']) <= .08 for b in rhythm)]
            if len(lead_refs) < 3 or len(backing_refs) < 3:
                continue
            a = _spatial_vector([spectrum(n, note['pitch']) for n in lead_refs])
            b = _spatial_vector([spectrum(n, note['pitch']) for n in backing_refs])
            if a is None or b is None:
                continue
            matrix = np.column_stack([a, b])
            condition = float(np.linalg.cond(matrix))
            if not np.isfinite(condition) or condition > 4.:
                continue
            target = dict(note, start=opts.grid_offset+cell*grid)
            projected = _tone_spectrum(data, sr, target) @ np.linalg.inv(matrix).T
            energy = np.sum(abs(projected)**2, axis=0)
            fraction = float(energy[0]/max(float(energy.sum()), 1e-15))
            if energy.sum() > 1e-12:
                models.append(dict(carrier_pitch=carrier, lead_fraction=fraction,
                                   condition=condition, lead_references=len(lead_refs),
                                   backing_references=len(backing_refs)))
        h['models'] = models
        if not models or min(m['lead_fraction'] for m in models) < .6:
            h['reason'] = 'ambiguous_or_backing_harmonic'
            continue
        if cell in choices and raw_notes[choices[cell]['raw_index']]['confidence'] >= note['confidence']:
            h['reason'] = 'competing_detection'
            continue
        if cell in choices:
            choices[cell]['reason'] = 'competing_detection'
        choices[cell] = h
    result_lead = [dict(n) for n in lead]
    result_rhythm = [dict(n) for n in rhythm]
    for cell, h in choices.items():
        note = raw_notes[h['raw_index']]
        added = arrange_melody([note], opts)
        if not added or step(added[0]) != cell:
            h['reason'] = 'outside_playable_or_entry_grid'
            continue
        displaced = [n for n in result_lead if step(n) == cell]
        removed = [n for n in result_rhythm if step(n) == cell
                   and n['pitch'] == note['pitch'] and n['confidence'] == note['confidence']]
        result_lead = [n for n in result_lead if step(n) != cell]+added
        result_rhythm = [n for n in result_rhythm if n not in removed]
        h.update(accepted=True, reason='motif_and_frequency_specific_lead',
                 added_lead=added[0], displaced_lead=displaced, removed_rhythm=removed)
    result_lead.sort(key=lambda n:n['start'])
    for a, b in zip(result_lead, result_lead[1:]):
        a['end'] = min(a['end'], b['start'])
    return result_lead, result_rhythm, hypotheses
