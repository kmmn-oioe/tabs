"""Conservative pitch-specific attack evidence for detected rhythm notes.

Scores describe measured energy, not trained instrument probabilities. An
ambiguous but strong fundamental is retained, including quiet legato notes.
"""
from __future__ import annotations

from math import ceil

import numpy as np

from .audio import load_audio_channels


def pitch_attack(data, sr, note):
    size = round(sr*.06)
    times = np.arange(max(0., note['start']-.18),
                      min(len(data)/sr, note['start']+.18), .01)
    if len(times) < 5:
        return dict(onset=note['start'], novelty=0., share=0., energy=0.)
    chunks = np.zeros((len(times), size, data.shape[1]))
    for i, time in enumerate(times):
        start = round(time*sr)-size//2
        lo, hi = max(0, start), min(len(data), start+size)
        if hi > lo:
            chunks[i, lo-start:hi-start] = data[lo:hi]
    spectrum = np.fft.rfft(chunks*np.hanning(size)[None, :, None], n=8192, axis=1)
    power = (abs(spectrum)**2).sum(axis=2)
    frequency = 440 * 2**((note['pitch']-69)/12)
    bins = np.fft.rfftfreq(8192, 1/sr)
    bands = np.array([power[:, abs(bins-h*frequency) < max(14, h*frequency*.018)].sum(axis=1)
                      for h in range(1, 6)]).T
    envelope = np.sqrt(bands[:, 0])
    rise = np.maximum(0., envelope-np.r_[envelope[:3], envelope[:-3]])
    indices = np.flatnonzero((times >= max(0., note['start']-.11))
                             & (times <= note['start']+.03))
    peak = indices[np.argmax(rise[indices])]
    return dict(onset=float(times[peak]),
                novelty=float(rise[peak]/max(envelope.max(), 1e-15)),
                share=float(bands[:, 0].max()/max(bands.sum(axis=1).max(), 1e-15)),
                energy=float(bands[:, 0].max()/max(power.sum(axis=1).max(), 1e-15)))


def analyze_rhythm_attacks(path, notes):
    data, sr = load_audio_channels(path)
    if not np.isfinite(data).all():
        raise ValueError('Audio contains non-finite samples')
    return [pitch_attack(data, sr, note) for note in notes]


def filter_rhythm_attacks(notes, attacks):
    if len(notes) != len(attacks):
        raise ValueError('Attack evidence must align with rhythm candidates')
    # Preserve low-amplitude notes: all quantities are relative, not loudness
    # thresholds. Strong fundamentals without a sharp attack remain ambiguous.
    return [dict(n) for n, a in zip(notes, attacks)
            if a['energy'] > 1e-8 and a['share'] >= .01
            and (a['novelty'] >= .05 or a['share'] >= .5)]


def rhythm_voice_checks(notes, features, lead_notes, lead_features):
    """Withhold a residual lead voice, requiring two stable stereo sources.

    Lead anchors must already have independent audio evidence. Pan alone,
    pitch height, or an isolated matching timbre cannot establish ownership.
    Weak fundamentals, mono and ambiguous mixtures are left undecided.
    """
    if len(notes) != len(features) or len(lead_notes) != len(lead_features):
        raise ValueError('Voice features must align with their notes')
    checks = []
    for note, feature in zip(notes, features):
        check = dict(reject=False, reason='ambiguous', pan=feature.pan,
                     lead_anchors=0, backing_anchors=0, attack=feature.attack)
        checks.append(check)
        if not feature.valid or feature.spatial_rank < .1:
            continue
        anchors = [(n, f) for n, f in zip(lead_notes, lead_features)
                   if f.valid and f.spatial_rank >= .1
                   and abs(n['start']-note['start']) <= 1.5
                   and abs(f.pan-feature.pan) <= .2
                   and float(np.dot(f.harmonics, feature.harmonics)) >= .8]
        check['lead_anchors'] = len(anchors)
        if (len({round(n['start']/.05) for n, f in anchors}) < 3
                or not any(n['start'] < note['start'] for n, f in anchors)
                or not any(n['start'] > note['start'] for n, f in anchors)):
            continue
        backing = [(n, f) for n, f in zip(notes, features)
                   if f.valid and f.spatial_rank >= .1
                   and abs(n['start']-note['start']) <= 1.5
                   and abs(f.pan-feature.pan) >= 1.4]
        check['backing_anchors'] = len(backing)
        if len({round(n['start']/.05) for n, f in backing}) < 3:
            continue
        pans = np.array([f.pan for n, f in backing])
        center = float(np.median(pans))
        check['backing_pan'] = center
        if float(np.median(abs(pans-center))) > .15:
            continue
        check.update(reject=True, reason='confirmed_lead_source',
                     lead_pan=float(np.median([f.pan for n, f in anchors])))
    return checks


def recover_lead_holes(lead, candidates, checks, opts):
    """Move a clearly rearticulated lead source into an empty melody cell.

    Never replace an existing lead attack or use ambiguous/held evidence to
    create one. Existing pitches, onsets and fingerings remain fixed.
    """
    from .tab import arrange_melody
    if len(candidates) != len(checks):
        raise ValueError('Voice checks must align with candidates')
    grid = opts.quantize*60/max(opts.bpm, 1)
    cutoff = max(0, ceil((opts.lead_start_seconds-opts.grid_offset)/grid))
    step = lambda n: max(cutoff, round((n['start']-opts.grid_offset)/grid))
    occupied = {step(n) for n in lead}
    choices = {}
    for note, check in zip(candidates, checks):
        cell = step(note)
        if (not check['reject'] or check['attack'] < .6 or cell in occupied
                or note['start'] < opts.lead_start_seconds):
            continue
        if cell not in choices or note['confidence'] > choices[cell]['confidence']:
            choices[cell] = note
    recovered = []
    for note in choices.values():
        clean = {k:v for k,v in note.items() if k not in ('_snap_start', '_audio_rejection')}
        recovered.extend(arrange_melody([clean], opts))
    combined = sorted([dict(n) for n in lead]+recovered, key=lambda n:n['start'])
    for current, following in zip(combined, combined[1:]):
        current['end'] = min(current['end'], following['start'])
    return combined, recovered


def align_rhythm_grid(notes, opts):
    """Estimate a small systematic detector delay from recurring onset phase.

    Do not replace the user's tempo/grid origin or invent attacks. Only a
    sufficiently populated, concentrated phase distribution can adjust snap
    decisions. Independent notes remain on their requested subdivision grid.
    """
    grid = opts.quantize*60/max(opts.bpm, 1)
    if grid <= 0:
        raise ValueError('Quantization must be positive')
    info = dict(applied=False, delay_seconds=0., concentration=0.)
    if len(notes) < 16 or len({round(n['start']/grid) for n in notes}) < 8:
        return [dict(n) for n in notes], info
    phases = np.array([(n['start']-opts.grid_offset)/grid for n in notes])
    vector = np.exp(2j*np.pi*phases).mean()
    concentration = float(abs(vector))
    delay = float(np.angle(vector)/(2*np.pi)*grid)
    info.update(concentration=concentration)
    # A fractional shift larger than half a grid or a weak/multimodal phase
    # may be a real rhythmic pattern. Leave those performances unchanged.
    if concentration < .45 or not .005 <= delay <= .45*grid:
        return [dict(n) for n in notes], info
    info.update(applied=True, delay_seconds=delay)
    return [dict(n, _snap_start=max(0., n['start']-delay)) for n in notes], info
