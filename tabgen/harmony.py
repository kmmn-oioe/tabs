"""Audio chroma and temporally smoothed, beat-aligned chord inference."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .audio import load_audio
from .rhythm import ChordChange, QUALITIES, _rank_chords
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


def _decode_chord_segments(observations, prior, opts, beat):
    """Semi-Markov decoding: jointly choose chord identities and their durations.

    Pooling evidence within each candidate span lets arpeggios describe one
    chord. Silence prevents a chord span from crossing a real rest; ambiguous
    audible frames do not automatically force a new N.C. segment.
    """
    labels = [(None, "", ())] + [(root, suffix, intervals)
              for root in range(12) for suffix, intervals in QUALITIES]
    lookup = {label: i for i, label in enumerate(labels)}
    size, count = len(labels), len(observations)
    times = [observations[0][0]] + [o[1] for o in observations]
    durations = np.diff(times) / beat
    chroma = np.array([o[2] for o in observations])
    bass = np.array([o[3] for o in observations])
    silent = np.array([o[4] for o in observations])
    chroma_sum = np.vstack([np.zeros(12), np.cumsum(chroma * durations[:, None], axis=0)])
    bass_sum = np.vstack([np.zeros(12), np.cumsum(bass * durations[:, None], axis=0)])
    silence_sum = np.r_[0, np.cumsum(silent)]
    active_duration = np.r_[0., np.cumsum(durations * ~silent)]
    # Aggregate chroma alone can hide real changes: two different chords can
    # share many tones. Also score every beat against the proposed held chord.
    # Missing tones are soft evidence, so an arpeggio need not sound all notes
    # simultaneously, but a sustained incompatible tone still counts against it.
    local = np.zeros((count, size))
    for i, (_, _, weights, lows, _) in enumerate(observations):
        vector = .75*_normalize(weights**2) + .25*_normalize(lows**2)
        vector = np.where(vector >= np.max(vector)*.12, vector, 0.)
        vector = _normalize(vector)
        for k, (root, _, intervals) in enumerate(labels[1:], 1):
            pcs = [(root+x) % 12 for x in intervals]
            inside = float(np.sum(vector[pcs]))
            missing = float(np.mean(vector[pcs] == 0))
            local[i,k] = (inside - 1.2*(1-inside) - .8*missing - .025*len(intervals)
                          + .28*float(lows[root]) + .03*float(np.mean(prior[pcs])))
    local_sum = np.vstack([np.zeros(size), np.cumsum(local*durations[:,None], axis=0)])

    switch = .25 + .15*opts.chord_beats
    costs = np.full((size, size), switch)
    for i, (root_a, _, intervals_a) in enumerate(labels):
        for j, (root_b, _, intervals_b) in enumerate(labels):
            if i == j:
                costs[i,j] = 0.
            elif root_a is not None and root_b is not None:
                a = {(root_a+x) % 12 for x in intervals_a}
                b = {(root_b+x) % 12 for x in intervals_b}
                # Require more evidence to toggle related chord colors.
                if len(a & b) >= 3 or root_a == root_b:
                    costs[i,j] += .15

    scores = np.full((count+1, size), -np.inf)
    scores[0] = 0.
    previous_time = np.zeros((count+1, size), dtype=int)
    previous_label = np.zeros((count+1, size), dtype=int)
    transitions = {0: (np.zeros(size), np.zeros(size, dtype=int))}
    for end in range(1, count+1):
        # Sixteen beats bounds cost on long songs; adjacent identical spans are
        # merged after decoding, so there is no maximum held-chord duration.
        for start in range(max(0, end-16), end):
            length = (times[end]-times[start])/beat
            emission = np.full(size, -np.inf)
            emission[0] = .12 * (active_duration[end]-active_duration[start])
            if silence_sum[end] == silence_sum[start]:
                weights = (chroma_sum[end]-chroma_sum[start])/length
                lows = (bass_sum[end]-bass_sum[start])/length
                vector = .75*_normalize(weights**2) + .25*_normalize(lows**2)
                ranked = _rank_chords(dict(enumerate(vector)), dict(enumerate(lows)))
                for score, root, suffix, intervals in ranked:
                    if score < .35:
                        continue
                    pcs = [(root+x) % 12 for x in intervals]
                    # Low-register support anchors the harmonic foundation;
                    # high ornaments alone should not keep changing the root.
                    score += .20*float(lows[root]) + .03*float(np.mean(prior[pcs]))
                    short = .15*max(0., opts.chord_beats-length)
                    k = lookup[root,suffix,intervals]
                    emission[k] = .2*score*length + .8*(local_sum[end,k]-local_sum[start,k])-short
            if start not in transitions:
                entries = scores[start][:, None]-costs
                parents = np.argmax(entries, axis=0)
                transitions[start] = entries[parents, np.arange(size)], parents
            entry_scores, parents = transitions[start]
            candidate = entry_scores+emission
            better = candidate > scores[end]
            scores[end,better] = candidate[better]
            previous_time[end,better] = start
            previous_label[end,better] = parents[better]

    changes = []
    end, label = count, int(np.argmax(scores[-1]))
    while end:
        start = previous_time[end,label]
        root, suffix, intervals = labels[label]
        changes.append(ChordChange(times[start], times[end], root, suffix, intervals))
        end, label = int(start), int(previous_label[end,label])
    result = []
    for change in reversed(changes):
        if result and result[-1].name == change.name:
            result[-1] = replace(result[-1], end=change.end)
        else:
            result.append(change)
    return result


def chords_from_chroma(features: ChromaFeatures, opts: TabOptions) -> list[ChordChange]:
    """Choose sustained harmony from variable-length spans on the beat grid."""
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
            frames[np.argmin(abs(features.times - (start + end) / 2))] = True
        weights = _normalize(np.mean(features.chroma[:, frames], axis=1))
        bass = _normalize(np.mean(features.bass[:, frames], axis=1))
        is_silent = float(np.mean(features.rms[frames])) < silence
        observations.append((start, end, weights, bass, is_silent))
    return _decode_chord_segments(observations, prior, opts, beat)


def audio_chord_changes(path: str, opts: TabOptions) -> list[ChordChange]:
    return chords_from_chroma(audio_chroma(path), opts)
