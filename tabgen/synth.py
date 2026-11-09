"""Render the final TAB notes over the guitar-free accompaniment."""
from __future__ import annotations

import math

import numpy as np
import soundfile as sf

from .audio import read_preview_audio


def render_tab_mix(backing_path: str, parts: list[list[dict]], output_path: str) -> str:
    """Mix only no-guitar audio and canonical, already arranged TAB notes.

    This uses an internal additive synthesizer, without a MIDI device or
    SoundFont. All voices share the accompaniment's absolute sample clock.
    """
    backing, sr = read_preview_audio(backing_path)
    if not len(backing) or sr <= 0 or backing.shape[1] not in (1, 2):
        raise ValueError("ギターオフ音源が空、またはチャンネル数が不正です")
    if not np.isfinite(backing).all():
        raise ValueError("ギターオフ音源に不正な値があります")
    notes = [n for part in parts for n in part]
    for n in notes:
        if (not all(math.isfinite(n[k]) for k in ('start', 'end', 'pitch'))
                or not 0 <= n['pitch'] <= 127 or n['end'] <= n['start']):
            raise ValueError("TABの発音情報が不正です")
    frames = max(len(backing), math.ceil(max((n['end'] for n in notes), default=0.) * sr))
    mix = np.zeros((frames, backing.shape[1]), dtype=np.float32)
    mix[:len(backing)] = backing * .8
    for part_index, part in enumerate(parts):
        gain = .18 if part_index == 0 else .13
        for n in part:
            first = max(0, round(n['start'] * sr))
            last = min(frames, round(n['end'] * sr))
            if last <= first:
                continue
            # Compute only this note's span, keeping long songs inexpensive.
            times = np.arange(last-first, dtype=np.float64) / sr
            phase = 2 * np.pi * (440 * 2 ** ((n['pitch']-69)/12)) * times
            wave = np.sin(phase)
            frequency = 440 * 2 ** ((n['pitch']-69)/12)
            for harmonic, level in ((2, .25), (3, .1)):
                if harmonic * frequency < sr/2:
                    wave += level * np.sin(harmonic * phase)
            # Short ramps end at note-off; rests stay silent and no oscillator
            # keeps running after a part has stopped.
            attack = min(.005, (last-first)/(2*sr))
            release = min(.015, (last-first)/(2*sr))
            envelope = np.minimum(1., times / max(attack, 1/sr))
            envelope *= np.minimum(1., (last-first-1-np.arange(last-first)) / max(release*sr, 1))
            envelope *= .8 + .2 * np.exp(-3*times)
            mix[first:last] += (gain * wave * envelope).astype(np.float32)[:, None]
    peak = float(np.max(np.abs(mix)))
    if peak > .98:
        mix *= .98/peak
    sf.write(output_path, mix, sr, subtype='FLOAT')
    return output_path
