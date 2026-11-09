"""Conservative pitch-specific attack evidence for detected rhythm notes.

Scores describe measured energy, not trained instrument probabilities. An
ambiguous but strong fundamental is retained, including quiet legato notes.
"""
from __future__ import annotations

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
