"""Track a monophonic melody through polyphonic note onsets.

Pitch register is a soft preference, not a lead/rhythm cutoff. This produces
an arrangement; it does not identify which physical guitar played a note.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import tanh, isfinite

DEFAULT_LEAD_THRESHOLD = 64


@dataclass
class _State:
    score: float
    pitch: int | None
    start: float
    parent: _State | None = None
    selected: int | None = None


def split_notes(notes: list[dict], lead_threshold: int = DEFAULT_LEAD_THRESHOLD,
                window: float = .04, min_confidence: float = .3, *,
                suppress_chords: bool = True, start_seconds: float = 0.,
                lead_scores: list[float] | None = None) -> tuple[list[dict], list[dict]]:
    """Choose a melodic path, returning copied lead notes and remaining notes.

    Near-simultaneous attacks are alternatives. A bounded dynamic program can
    skip weak notes, follows small intervals, and relaxes continuity after a
    rest. Confidence and duration support low-register melodies as well as
    high ones. Selected notes are released at the next melody attack.
    """
    if window <= 0:
        raise ValueError("Onset grouping window must be positive")
    if not isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Lead start must be a finite nonnegative number")
    if lead_scores is not None and len(lead_scores) != len(notes):
        raise ValueError("Audio evidence must align with the detected notes")
    ordered = sorted((i for i, n in enumerate(notes)
                      if n["confidence"] >= min_confidence and n["end"] > n["start"]
                      and n["start"] >= start_seconds
                      and (lead_scores is None or lead_scores[i] >= .6)),
                     key=lambda i: (notes[i]["start"], notes[i]["pitch"]))
    if suppress_chords:
        # Strummed chords have slightly staggered attacks: use a wider window
        # than the melody lattice, and require actual overlap to retain runs.
        clusters = []
        for i in ordered:
            if clusters and notes[i]["start"] - notes[clusters[-1][0]]["start"] < .09:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        blocked = set()
        for cluster in clusters:
            pitches = {notes[i]["pitch"] for i in cluster}
            overlap = min(notes[i]["end"] for i in cluster) - max(notes[i]["start"] for i in cluster)
            if len(pitches) >= 3 and len({p % 12 for p in pitches}) >= 2 and overlap >= .025:
                blocked.update(cluster)
        ordered = [i for i in ordered if i not in blocked]
    groups: list[list[int]] = []
    for i in ordered:
        n = notes[i]
        if groups and n["start"] - notes[groups[-1][0]]["start"] < window:
            groups[-1].append(i)
        else:
            groups.append([i])
    if not groups:
        return [], list(notes)

    states = [_State(0., None, 0.)]
    for group in groups:
        expanded = [_State(s.score, s.pitch, s.start, s) for s in states]
        # Duplicate detections of the same pitch should not compete as voices.
        by_pitch = {}
        for i in group:
            n = notes[i]
            if n["pitch"] not in by_pitch or n["confidence"] > notes[by_pitch[n["pitch"]]]["confidence"]:
                by_pitch[n["pitch"]] = i
        for i in by_pitch.values():
            n = notes[i]
            reward = (2 * (n["confidence"] - .2)
                      + .25 * min(.6, n["end"] - n["start"])
                      + .12 * tanh((n["pitch"] - lead_threshold) / 12))
            def continuation(s):
                if s.pitch is None:
                    return s.score
                gap = max(.1, n["start"] - s.start)
                interval = abs(n["pitch"] - s.pitch)
                movement = min(1.2, interval * .055 + max(0, interval - 7) * .08)
                return s.score - movement / max(1., gap * 2)
            previous = max(states, key=continuation)
            expanded.append(_State(continuation(previous) + reward, n["pitch"], n["start"], previous, i))
        # Keep the best history for each continuity state, including skipped onsets.
        best = {}
        for s in expanded:
            key = s.pitch, s.start
            if key not in best or s.score > best[key].score:
                best[key] = s
        states = sorted(best.values(), key=lambda s: s.score, reverse=True)[:64]

    state = max(states, key=lambda s: s.score)
    selected = []
    while state is not None:
        if state.selected is not None:
            selected.append(state.selected)
        state = state.parent
    selected.reverse()
    lead = [dict(notes[i]) for i in selected]
    for current, following in zip(lead, lead[1:]):
        current["end"] = min(current["end"], following["start"])
    chosen = set(selected)
    return lead, [n for i, n in enumerate(notes) if i not in chosen]
