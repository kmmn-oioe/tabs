"""Experimental audio evidence for a voice independent of the accompaniment.

This is a conservative, untrained source-evidence heuristic, not a calibrated
lead-guitar classifier. A lone arpeggio cannot establish a separate lead.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

from .audio import load_audio_channels


@dataclass
class NoteAudioFeatures:
    pan: float
    harmonics: np.ndarray
    attack: float
    coherence: float
    valid: bool


@dataclass
class LeadAudioEvidence:
    scores: list[float]
    stereo: bool
    reference_found: bool
    features: list[NoteAudioFeatures]

    def summary(self) -> str:
        supported = sum(s >= .6 for s in self.scores)
        channels = "ステレオ定位・倍音・発音" if self.stereo else "モノラル倍音・発音（定位情報なし）"
        reference = "伴奏の基準音色を推定しました。" if self.reference_found else "伴奏の基準音色が不足しています。"
        return (f"リード音声判定: {channels}\n"
                f"{reference}\n"
                f"独立した声部の根拠あり: {supported}/{len(self.scores)}音。"
                "根拠が弱い／判別不能な候補は休符にします。\n"
                "※ 実験的な証拠スコアです。学習済みの存在確率ではありません。")


def _partials(samples, sr, pitch):
    if len(samples) < 16:
        return np.zeros((5, samples.shape[1])), 0.
    n_fft = max(8192, 1 << (len(samples) - 1).bit_length())
    windowed = samples * np.hanning(len(samples))[:, None]
    power = abs(np.fft.rfft(windowed, n=n_fft, axis=0)) ** 2
    resolution = sr / n_fft
    f0 = 440 * 2 ** ((pitch - 69) / 12)
    partials = []
    for harmonic in range(1, 6):
        center = int(round(f0 * harmonic / resolution))
        radius = max(1, int(round(f0 * harmonic * .012 / resolution)))
        lo, hi = max(1, center - radius), min(len(power), center + radius + 1)
        if lo >= hi:
            partials.append(np.zeros(samples.shape[1]))
            continue
        # Local background subtraction reduces broadband/transient evidence.
        context = power[max(1, lo-2*radius-2):min(len(power), hi+2*radius+2)]
        floor = np.median(context, axis=0)
        partials.append(np.maximum(0., np.sum(power[lo:hi], axis=0) - (hi-lo)*floor))
    return np.array(partials), float(np.sum(power))


def note_audio_features(data, sr, note) -> NoteAudioFeatures:
    duration = min(.35, max(0., note["end"] - note["start"]))
    start = max(0, int((note["start"] + min(.035, duration*.15)) * sr))
    stop = max(start, min(len(data), int((note["start"] + duration) * sr)))
    partials, total = _partials(data[start:stop], sr, note["pitch"])
    energy = partials.sum(axis=1)
    harmonic_total = float(energy.sum())
    valid = (total > 1e-8 and harmonic_total / total > .01
             and energy[0] > harmonic_total * .04)
    amplitudes = np.sqrt(energy)
    signature = amplitudes / max(float(np.linalg.norm(amplitudes)), 1e-12)
    if data.shape[1] == 2:
        pans = (partials[:, 1] - partials[:, 0]) / np.maximum(partials.sum(axis=1), 1e-12)
        weights = energy / max(harmonic_total, 1e-12)
        pan = float(np.sum(pans * weights))
        spread = float(np.sqrt(np.sum(weights * (pans-pan)**2)))
        coherence = max(.3, 1 - spread)
    else:
        pan, coherence = 0., 1.
    before_start = max(0, int((note["start"] - .15) * sr))
    before_end = max(0, int((note["start"] - .02) * sr))
    before, _ = _partials(data[before_start:before_end], sr, note["pitch"])
    # Normalize FFT energy by squared window length for before/after comparison.
    after_level = float(energy[0]) / max(1, stop-start)**2
    before_level = float(before[0].sum()) / max(1, before_end-before_start)**2
    attack = float(np.clip(1 - before_level / max(after_level, 1e-12), 0, 1))
    return NoteAudioFeatures(pan, signature, attack, coherence, bool(valid))


def _distance(a, b):
    return abs(a.pan-b.pan)/.45 + float(np.linalg.norm(a.harmonics-b.harmonics))/.55


def _chord_anchors(notes):
    groups = []
    for i in sorted(range(len(notes)), key=lambda i: notes[i]["start"]):
        if groups and notes[i]["start"] - notes[groups[-1][0]]["start"] < .09:
            groups[-1].append(i)
        else:
            groups.append([i])
    return {i for group in groups
            if len({notes[j]["pitch"] for j in group}) >= 3
            and min(notes[j]["end"] for j in group) > max(notes[j]["start"] for j in group)+.025
            for i in group}


def infer_lead_evidence(notes, features, stereo) -> LeadAudioEvidence:
    """Compare persistent source signatures and require simultaneous backing.

    Chord attacks favor an accompaniment reference. Otherwise the most common
    stable signature is a tentative reference. A different pan/timbre alone is
    insufficient: its source must recur (or sustain), and overlap the reference.
    """
    anchors = _chord_anchors(notes)
    clusters = []
    membership = {}
    for i in sorted(range(len(notes)), key=lambda i: notes[i]["start"]):
        feature = features[i]
        if not feature.valid:
            continue
        if clusters:
            nearest = min(range(len(clusters)), key=lambda j: _distance(feature, features[clusters[j][0]]))
        else:
            nearest = None
        if nearest is None or (_distance(feature, features[clusters[nearest][0]]) > 1. and len(clusters) < 24):
            nearest = len(clusters)
            clusters.append([])
        clusters[nearest].append(i)
        membership[i] = nearest
    scores = [0.] * len(notes)
    if not clusters:
        return LeadAudioEvidence(scores, stereo, False, features)
    reference_index = max(range(len(clusters)),
                          key=lambda j: len(clusters[j]) + 6 * len(set(clusters[j]) & anchors))
    reference = clusters[reference_index]
    if len(reference) < 3:
        return LeadAudioEvidence(scores, stereo, False, features)
    reference_pan = float(np.median([features[i].pan for i in reference]))
    reference_timbre = np.median([features[i].harmonics for i in reference], axis=0)
    reference_timbre /= max(float(np.linalg.norm(reference_timbre)), 1e-12)
    reference_intervals = []
    for start, end in sorted((notes[i]["start"], notes[i]["end"]) for i in reference):
        if reference_intervals and start <= reference_intervals[-1][1]:
            reference_intervals[-1][1] = max(reference_intervals[-1][1], end)
        else:
            reference_intervals.append([start, end])
    starts = [start for start, _ in reference_intervals]
    cluster_times = [sorted(notes[i]["start"] for i in group) for group in clusters]
    for i, feature in enumerate(features):
        if i not in membership or membership[i] == reference_index:
            continue
        note = notes[i]
        index = max(0, bisect_right(starts, note["start"]) - 1)
        overlap = 0.
        for start, end in reference_intervals[index:]:
            if start >= note["end"]:
                break
            overlap += max(0., min(end, note["end"]) - max(start, note["start"]))
        if overlap < max(.04, .5 * (note["end"]-note["start"])):
            continue
        times = cluster_times[membership[i]]
        neighbors = bisect_right(times, note["start"]+1) - bisect_left(times, note["start"]-1)
        if neighbors < 3 and note["end"] - note["start"] < .6:
            continue
        pan = float(np.clip((abs(feature.pan-reference_pan)-.12)/.4, 0, 1)) if stereo else 0.
        timbre = float(np.clip((np.linalg.norm(feature.harmonics-reference_timbre)-.2)/.45, 0, 1))
        difference = max(pan, .8*timbre)
        articulation = .65 + .2*feature.attack + .15*min(1., (note["end"]-note["start"])/.2)
        scores[i] = float(np.clip(difference * articulation * feature.coherence, 0, 1))
    # Require repeated evidence of *simultaneous* voices, rather than just a
    # changing pan or one short crossfade at the end of an accompaniment note.
    supported_times = [sorted(notes[i]["start"] for i in group if scores[i] >= .6) for group in clusters]
    for i, score in enumerate(scores):
        if score < .6:
            continue
        times = supported_times[membership[i]]
        local = bisect_right(times, notes[i]["start"]+1) - bisect_left(times, notes[i]["start"]-1)
        if local < 3 and notes[i]["end"] - notes[i]["start"] < .6:
            scores[i] = 0.
    return LeadAudioEvidence(scores, stereo, True, features)


def analyze_lead_audio(path: str, notes: list[dict], *, min_confidence: float = .3) -> LeadAudioEvidence:
    data, sr = load_audio_channels(path)
    if data.shape[1] not in (1, 2):
        raise ValueError("Lead audio analysis supports mono or stereo files")
    features = [note_audio_features(data, sr, n) if n["confidence"] >= min_confidence
                else NoteAudioFeatures(0., np.zeros(5), 0., 0., False) for n in notes]
    return infer_lead_evidence(notes, features, data.shape[1] == 2)
