"""Split transcribed notes into lead and rhythm guitar parts.

Heuristic: at each short time window, the top voice (or a high single note)
is treated as lead; everything else is rhythm. Uses a pitch threshold and
density cues — no extra model required.
"""
from __future__ import annotations

from .tab import TabOptions

DEFAULT_LEAD_THRESHOLD = 64  # MIDI E4


def split_notes(notes: list[dict], lead_threshold: int = DEFAULT_LEAD_THRESHOLD,
                window: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Return (lead_notes, rhythm_notes)."""
    if not notes:
        return [], []

    lead: list[dict] = []
    rhythm: list[dict] = []

    # group notes by overlapping time windows
    windows: list[list[dict]] = []
    for n in sorted(notes, key=lambda x: x["start"]):
        if windows and n["start"] - windows[-1][0]["start"] < window:
            windows[-1].append(n)
        else:
            windows.append([n])

    for w in windows:
        pitches = [n["pitch"] for n in w]
        top = max(pitches)
        for n in w:
            if n["pitch"] == top and (len(w) > 1 or top >= lead_threshold):
                lead.append(n)
            elif n["pitch"] >= lead_threshold and len(w) == 1:
                lead.append(n)
            else:
                rhythm.append(n)
    return lead, rhythm
