"""Standard music notation export (MusicXML) from transcribed notes."""
from __future__ import annotations

from .tab import TabOptions


def export_musicxml(notes: list[dict], path: str, opts: TabOptions | None = None,
                    title: str = "tabs"):
    """Write notes as a MusicXML file readable by MuseScore / Finale / etc."""
    from music21 import (stream, note as m21note, chord as m21chord,
                         meter, tempo, metadata, harmony)

    opts = opts or TabOptions()
    sec_per_beat = 60.0 / max(opts.bpm, 1.0)

    # group simultaneous notes into chords (tolerance 30 ms)
    events: list[tuple[float, float, list[int]]] = []
    for n in sorted(notes, key=lambda x: x["start"]):
        if events and abs(n["start"] - events[-1][0]) < 0.03:
            events[-1][2].append(n["pitch"])
            events[-1] = (events[-1][0], max(events[-1][1], n["end"]), events[-1][2])
        else:
            events.append((n["start"], n["end"], [n["pitch"]]))

    # build a flat part; music21's makeMeasures handles bars/rests/notation
    part = stream.Part()
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, tempo.MetronomeMark(number=int(opts.bpm)))
    if title:
        part.insert(0, metadata.Metadata(title=title))

    symbols = {(n["start"], n["chord"]) for n in notes if n.get("chord")}
    for start, symbol in sorted(symbols):
        part.insert(round(start / sec_per_beat * 4) / 4, harmony.ChordSymbol(symbol))

    for i, (start, end, pitches) in enumerate(events):
        start_beat = round(start / sec_per_beat * 4) / 4
        next_start = events[i + 1][0] if i + 1 < len(events) else end
        dur_beats = max(0.25, round((min(end, next_start) - start) / sec_per_beat * 4) / 4)
        pitches = sorted(set(pitches))
        el = (m21chord.Chord(pitches) if len(pitches) > 1
              else m21note.Note(pitches[0]))
        el.quarterLength = dur_beats
        try:
            part.insert(start_beat, el)
        except Exception:
            part.insert(part.highestTime, el)  # overlap safety

    part.makeMeasures(inPlace=True)
    part.makeNotation(inPlace=True)

    sc = stream.Score()
    sc.insert(0, part)
    sc.write("musicxml", fp=path)
