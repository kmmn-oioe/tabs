"""Export helpers: text tab, MIDI."""
from __future__ import annotations

import pretty_midi

from .tab import TabOptions


def export_midi(notes: list[dict], path: str):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=25, is_drum=False)  # acoustic guitar
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=n["pitch"], start=n["start"], end=max(n["end"], n["start"] + 0.05)
        ))
    pm.instruments.append(inst)
    pm.write(path)
