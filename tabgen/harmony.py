"""Audio chroma and temporally smoothed, beat-aligned chord inference."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .audio import load_audio
from .rhythm import ChordChange, _rank_chords
from .tab import TabOptions


@dataclass
class ChromaFeatures:
    times: np.ndarray
    chroma: np.ndarray       # shape (12, frames)
    bass: np.ndarray         # shape (12, frames)
    rms: np.ndarray
    duration: float


def audio_chroma(path: str) -> ChromaFeatures:
    """Suppress transient energy and extract harmonic and bass pitch classes.

    HPSS is a harmonic/percussive mask, not guitar/vocal source separation.
    No additional models or network downloads are needed.
    """
    import librosa
    from scipy.ndimage import median_filter

    mono, sr = load_audio(path, sr=22050)
    duration = len(mono) / sr
    if not len(mono):
        return ChromaFeatures(np.empty(0), np.empty((12, 0)), np.empty((12, 0)), np.empty(0), 0.)
    n_fft, hop = 8192, 1024
    # Explicit padding avoids short-input FFT warnings; frame centers stay at t=0.
    padded = np.pad(mono, (n_fft // 2, n_fft // 2))
    spectrum = librosa.stft(padded, n_fft=n_fft, hop_length=hop, center=False)
    power = np.abs(spectrum) ** 2
    del spectrum
    horizontal = median_filter(power, size=(1, 9))
    vertical = median_filter(power, size=(17, 1))
    mask = horizontal ** 2 / (horizontal ** 2 + vertical ** 2 + 1e-12)
    harmonic = power * mask
    chroma = librosa.feature.chroma_stft(S=harmonic, sr=sr, n_fft=n_fft,
                                        hop_length=hop, tuning=0., norm=None)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low = harmonic * np.exp(-frequencies[:, None] / 350)
    bass = librosa.feature.chroma_stft(S=low, sr=sr, n_fft=n_fft,
                                      hop_length=hop, tuning=0., norm=None)
    rms = librosa.feature.rms(y=padded, frame_length=n_fft, hop_length=hop, center=False)[0]
    times = np.arange(chroma.shape[1]) * hop / sr
    keep = times < duration
    return ChromaFeatures(times[keep], chroma[:, keep], bass[:, keep], rms[keep], duration)


def _normalize(vector):
    return vector / max(float(np.sum(vector)), 1e-12)


def _key_prior(chroma):
    # Krumhansl major/minor profiles provide a weak global context only.
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    centered = chroma - np.mean(chroma)
    if np.linalg.norm(centered) < 1e-9:
        return np.zeros(12)
    candidates = []
    for profile in (major, minor):
        profile = (profile - profile.mean()) / profile.std()
        for root in range(12):
            rotated = np.roll(profile, root)
            candidates.append((float(np.dot(centered, rotated) / np.linalg.norm(centered)), rotated))
    candidates.sort(key=lambda item: item[0], reverse=True)
    confidence = min(1., max(0., candidates[0][0] - candidates[1][0]) * 3)
    return candidates[0][1] * confidence


def chords_from_chroma(features: ChromaFeatures, opts: TabOptions) -> list[ChordChange]:
    """Decode candidates jointly over beats, allowing changes inside old 2/4-beat blocks.

    Silence and unsupported single-pitch regions force N.C. The transition cost
    falls when adjacent chroma vectors change, while a weak key prior and
    persistence cost discourage flickering on passing tones.
    """
    if opts.chord_beats not in (1, 2, 4):
        raise ValueError("Chord stability must be 1, 2 or 4 beats")
    if features.duration <= 0 or not features.times.size:
        return []
    beat = 60 / max(opts.bpm, 1)
    grid = np.arange(opts.grid_offset, features.duration, beat)
    boundaries = sorted({0., features.duration, *(float(t) for t in grid if 0 < t < features.duration)})
    silence = max(1e-5, float(np.max(features.rms)) * .015)
    prior = _key_prior(np.sum(features.chroma, axis=1))
    observations = []
    for start, end in zip(boundaries, boundaries[1:]):
        frames = (features.times >= start) & (features.times < end)
        if not np.any(frames):
            # Very short pickup/tail intervals use the closest measured frame.
            frames[np.argmin(abs(features.times - (start + end) / 2))] = True
        weights = np.mean(features.chroma[:, frames], axis=1)
        bass = np.mean(features.bass[:, frames], axis=1)
        # Chroma filters spread energy into neighboring semitones. Sharpen the
        # peaks so C's spectral skirt is not mistaken for a major seventh (B).
        vector = .75 * _normalize(weights ** 2) + .25 * _normalize(bass ** 2)
        candidates = []
        if float(np.mean(features.rms[frames])) >= silence:
            ranked = _rank_chords(dict(enumerate(vector)), dict(enumerate(bass)))
            for score, root, suffix, intervals in ranked:
                if score < .35:
                    continue
                pcs = [(root + i) % 12 for i in intervals]
                candidates.append((score + .03 * float(np.mean(prior[pcs])), root, suffix, intervals))
                if len(candidates) == 8:
                    break
        # An explicit unharmonized option prevents weak evidence forcing a chord.
        candidates.append((.35, None, "", ()))
        observations.append((start, end, vector, candidates))

    layers = []
    for i, (start, end, vector, candidates) in enumerate(observations):
        layer = []
        for emission, root, suffix, intervals in candidates:
            if i:
                previous_vector = observations[i - 1][2]
                norm = np.linalg.norm(vector) * np.linalg.norm(previous_vector)
                difference = 1 - float(np.dot(vector, previous_vector) / max(norm, 1e-12))
                switch_cost = (.10 + .035 * opts.chord_beats) * max(.25, 1 - 3 * difference)
                best, parent = max((node[0] - (switch_cost if (node[1], node[2]) != (root, suffix) else 0), j)
                                   for j, node in enumerate(layers[-1]))
            else:
                best, parent = 0., -1
            layer.append((best + emission * min(1., (end - start) / beat), root, suffix, intervals, parent))
        layers.append(layer)
    chosen = max(range(len(layers[-1])), key=lambda j: layers[-1][j][0])
    path = []
    for layer in reversed(layers):
        node = layer[chosen]
        path.append(node)
        chosen = node[4]
    changes = []
    for (start, end, _, _), (_, root, suffix, intervals, _) in zip(observations, reversed(path)):
        change = ChordChange(start, end, root, suffix, intervals)
        if changes and changes[-1].name == change.name:
            changes[-1] = replace(changes[-1], end=end)
        else:
            changes.append(change)
    return changes


def audio_chord_changes(path: str, opts: TabOptions) -> list[ChordChange]:
    return chords_from_chroma(audio_chroma(path), opts)
