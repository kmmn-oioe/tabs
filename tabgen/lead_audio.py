"""Experimental audio evidence for a voice independent of the accompaniment.

This is a conservative, untrained source-evidence heuristic, not a calibrated
lead-guitar classifier. A lone arpeggio cannot establish a separate lead.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
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
    side_fraction: float = 0.
    spatial_rank: float = 1.


@dataclass
class LeadAudioEvidence:
    scores: list[float]
    stereo: bool
    reference_found: bool
    features: list[NoteAudioFeatures]
    method: str = "reference"

    def summary(self) -> str:
        supported = sum(s >= .6 for s in self.scores)
        channels = "ステレオ定位・位相・倍音・発音" if self.stereo else "モノラル倍音・発音（定位情報なし）"
        reference = ("定位が異なる声部の重なりと旋律の動きを確認しました。" if self.method == "spatial_phrase"
                     else "伴奏の基準音色を推定しました。" if self.reference_found
                     else "伴奏の基準音色が不足しています。")
        return (f"リード音声判定: {channels}\n"
                f"{reference}\n"
                f"独立した声部の根拠あり: {supported}/{len(self.scores)}音。"
                "根拠が弱い／判別不能な候補は休符にします。\n"
                "※ 実験的な証拠スコアです。学習済みの存在確率ではありません。")


def _partials(samples, sr, pitch):
    if len(samples) < 16:
        return np.zeros((5, samples.shape[1])), 0., np.zeros(5)
    n_fft = max(8192, 1 << (len(samples) - 1).bit_length())
    windowed = samples * np.hanning(len(samples))[:, None]
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=0)
    power = abs(spectrum) ** 2
    side_power = abs((spectrum[:, 0]-spectrum[:, 1])/np.sqrt(2))**2 if samples.shape[1] == 2 else None
    resolution = sr / n_fft
    f0 = 440 * 2 ** ((pitch - 69) / 12)
    partials, sides = [], []
    for harmonic in range(1, 6):
        center = int(round(f0 * harmonic / resolution))
        radius = max(1, int(round(f0 * harmonic * .012 / resolution)))
        lo, hi = max(1, center - radius), min(len(power), center + radius + 1)
        if lo >= hi:
            partials.append(np.zeros(samples.shape[1]))
            sides.append(0.)
            continue
        # Local background subtraction reduces broadband/transient evidence.
        context = power[max(1, lo-2*radius-2):min(len(power), hi+2*radius+2)]
        floor = np.median(context, axis=0)
        partials.append(np.maximum(0., np.sum(power[lo:hi], axis=0) - (hi-lo)*floor))
        if side_power is not None:
            side_context = side_power[max(1, lo-2*radius-2):min(len(power), hi+2*radius+2)]
            sides.append(max(0., float(np.sum(side_power[lo:hi]) - (hi-lo)*np.median(side_context))))
        else:
            sides.append(0.)
    return np.array(partials), float(np.sum(power)), np.array(sides)


def note_audio_features(data, sr, note) -> NoteAudioFeatures:
    duration = min(.35, max(0., note["end"] - note["start"]))
    start = max(0, int((note["start"] + min(.035, duration*.15)) * sr))
    stop = max(start, min(len(data), int((note["start"] + duration) * sr)))
    partials, total, sides = _partials(data[start:stop], sr, note["pitch"])
    energy = partials.sum(axis=1)
    harmonic_total = float(energy.sum())
    valid = (total > 1e-8 and harmonic_total / total > .01
             and energy[0] > harmonic_total * .04)
    amplitudes = np.sqrt(energy)
    signature = amplitudes / max(float(np.linalg.norm(amplitudes)), 1e-12)
    if data.shape[1] == 2:
        pans = (partials[:, 1] - partials[:, 0]) / np.maximum(partials.sum(axis=1), 1e-12)
        # Overlapping fundamentals can contaminate upper partials. Give the
        # detected fundamental the strongest vote in its spatial fingerprint.
        weights = .4 * energy / max(harmonic_total, 1e-12)
        weights[0] += .6
        pan = float(np.sum(pans * weights))
        spread = float(np.sqrt(np.sum(weights * (pans-pan)**2)))
        coherence = max(.3, 1 - spread)
        side_fraction = float(np.sum(weights * np.clip(sides / np.maximum(energy, 1e-12), 0, 1)))
        chunk = data[start:stop].astype(np.float64)
        covariance = chunk.T @ chunk / max(1, len(chunk))
        eigenvalues = np.linalg.eigvalsh(covariance)
        spatial_rank = float(2*max(0., eigenvalues[0])/max(float(eigenvalues.sum()), 1e-12))
    else:
        pan, coherence = 0., 1.
        side_fraction, spatial_rank = 0., 0.
    before_start = max(0, int((note["start"] - .15) * sr))
    before_end = max(0, int((note["start"] - .02) * sr))
    before, _, _ = _partials(data[before_start:before_end], sr, note["pitch"])
    # Normalize FFT energy by squared window length for before/after comparison.
    after_level = float(energy[0]) / max(1, stop-start)**2
    before_level = float(before[0].sum()) / max(1, before_end-before_start)**2
    attack = float(np.clip(1 - before_level / max(after_level, 1e-12), 0, 1))
    return NoteAudioFeatures(pan, signature, attack, coherence, bool(valid), side_fraction, spatial_rank)


def _distance(a, b, stereo=True):
    timbre = float(np.linalg.norm(a.harmonics-b.harmonics))/.55
    if not stereo:
        return timbre
    # Guitar timbre changes with pitch/chord and must not create a new source
    # on every voicing. Phase and pan define the main stereo source identity.
    return max(abs(a.pan-b.pan)/.45, abs(a.side_fraction-b.side_fraction)/.3) + .1*timbre


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


def _opposed_voice_evidence(notes, features):
    """Resolve a clear melodic stream among two persistent stereo sources.

    A frequent melody must not automatically become the backing reference.
    Use repeated simultaneous sources and phrase motion instead of choosing
    whichever spatial cluster contains most detections. Ambiguous/centered
    scenes retain the more conservative reference-based analysis below.
    """
    groups = []
    sorted_pans, centers = [], []
    for i in sorted(range(len(notes)), key=lambda i: notes[i]["start"]):
        if not features[i].valid or features[i].spatial_rank < .1:
            # This more permissive phrase route needs a measurable second
            # channel component. Quiet/ambiguous scenes may still use the
            # original phase/timbre evidence; do not extrapolate a later
            # stereo voice into a nearly rank-one intro.
            continue
        nearest = min(range(len(groups)), key=lambda j: abs(features[i].pan-centers[j])) if groups else None
        if nearest is None or abs(features[i].pan-centers[nearest]) > .32:
            groups.append([i])
            sorted_pans.append([features[i].pan])
            centers.append(features[i].pan)
        else:
            groups[nearest].append(i)
            pans = sorted_pans[nearest]
            insort(pans, features[i].pan)
            centers[nearest] = (pans[(len(pans)-1)//2] + pans[len(pans)//2])/2
    sources = []
    for group in groups:
        if len(group) < 3:
            continue
        pans = np.array([features[i].pan for i in group])
        center = float(np.median(pans))
        # A mixed center channel with unstable fingerprints is not a source.
        if float(np.median(abs(pans-center))) > .13:
            continue
        pitches = np.array([notes[i]["pitch"] for i in group])
        weights = np.array([notes[i]["confidence"] for i in group])
        histogram = np.bincount(pitches % 12, weights=weights, minlength=12)
        diversity = 1-float(histogram.max()/max(histogram.sum(), 1e-12))
        movement = np.abs(np.diff(pitches))
        motion = float(np.mean((movement > 0) & (movement % 12 != 0) & (movement <= 12)))
        role = .6*diversity + .4*motion
        sources.append((role, center, group))
    if len(sources) < 2:
        return None
    sources.sort(reverse=True, key=lambda s: s[0])
    role, center, melody = sources[0]
    backing = [s for s in sources[1:] if abs(s[1]-center) >= .65]
    if not backing or role < .4 or role-max(s[0] for s in backing) < .12:
        return None
    backing_notes = [j for _, _, group in backing for j in group]
    intervals = []
    for start, end in sorted((notes[i]['start'], notes[i]['end']) for i in backing_notes):
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    starts = [start for start, _ in intervals]
    scores = [0.] * len(notes)
    supported = []
    for i in melody:
        n = notes[i]
        at = max(0, bisect_right(starts, n['start'])-1)
        overlap = False
        for start, end in intervals[at:]:
            if start >= n['end']:
                break
            if min(n['end'], end)-max(n['start'], start) >= .04:
                overlap = True
                break
        if overlap:
            supported.append(i)
    # A single source that changes pan creates only a brief crossfade, not
    # repeated overlapping attacks from two independently recurring streams.
    times = sorted(notes[i]['start'] for i in supported)
    for i in supported:
        neighbors = bisect_right(times, notes[i]['start']+1)-bisect_left(times, notes[i]['start']-1)
        if neighbors >= 3:
            scores[i] = .6 + .25*features[i].attack + .15*features[i].coherence
    return scores if sum(s >= .6 for s in scores) >= 3 else None


def _reference_lead_evidence(notes, features, stereo) -> LeadAudioEvidence:
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
            nearest = min(range(len(clusters)), key=lambda j: _distance(feature, features[clusters[j][0]], stereo))
        else:
            nearest = None
        if nearest is None or (_distance(feature, features[clusters[nearest][0]], stereo) > 1. and len(clusters) < 24):
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
    reference_side = float(np.median([features[i].side_fraction for i in reference]))
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
        phase = float(np.clip((abs(feature.side_fraction-reference_side)-.1)/.35, 0, 1)) if stereo else 0.
        timbre = float(np.clip((np.linalg.norm(feature.harmonics-reference_timbre)-.2)/.45, 0, 1))
        if stereo:
            # Timbre alone changes with fingering and creates false sources.
            # Spatial rank is supporting evidence, never a hard veto: a quiet
            # real voice can have little effect on the full-band covariance.
            channel_support = .85 + .15*float(np.clip(feature.spatial_rank/.2, 0, 1))
            difference = max(pan, phase) * channel_support
        else:
            difference = .8*timbre
        articulation = .65 + .2*feature.attack + .15*min(1., (note["end"]-note["start"])/.2)
        scores[i] = float(np.clip(difference * articulation * (.7+.3*feature.coherence), 0, 1))
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


def infer_lead_evidence(notes, features, stereo) -> LeadAudioEvidence:
    """Follow local stereo phrases instead of assigning one role for a song.

    Each four-second section has two seconds of context on either side. Only
    a clear recurring stereo phrase overrides the conservative reference
    evidence, including its zero scores for simultaneous accompaniment notes.
    """
    reference = _reference_lead_evidence(notes, features, stereo)
    if not stereo or not notes:
        return reference
    scores = list(reference.scores)
    ordered = sorted(range(len(notes)), key=lambda i: notes[i]['start'])
    starts = [notes[i]['start'] for i in ordered]
    applied = False
    for first in np.arange(0., starts[-1]+.001, 4.):
        lo, hi = bisect_left(starts,max(0.,first-2)), bisect_left(starts,first+6)
        indices = ordered[lo:hi]
        local = _opposed_voice_evidence([notes[i] for i in indices], [features[i] for i in indices])
        if local is None:
            continue
        applied = True
        for i,score in zip(indices,local):
            if first <= notes[i]['start'] < first+4 and features[i].spatial_rank >= .1:
                scores[i] = score
    return LeadAudioEvidence(scores,stereo,reference.reference_found or applied,features,
                             "spatial_phrase" if applied else reference.method)


def analyze_lead_audio(path: str, notes: list[dict], *, min_confidence: float = .3) -> LeadAudioEvidence:
    data, sr = load_audio_channels(path)
    if data.shape[1] not in (1, 2):
        raise ValueError("Lead audio analysis supports mono or stereo files")
    features = [note_audio_features(data, sr, n) if n["confidence"] >= min_confidence
                else NoteAudioFeatures(0., np.zeros(5), 0., 0., False) for n in notes]
    return infer_lead_evidence(notes, features, data.shape[1] == 2)
