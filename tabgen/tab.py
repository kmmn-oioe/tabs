"""Convert transcribed notes into a guitar tablature."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

# Standard tuning, string 1 (high E) ... string 6 (low E)
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # MIDI pitches of open strings


@dataclass
class TabOptions:
    tuning: list[int] = field(default_factory=lambda: list(STANDARD_TUNING))
    quantize: float = 0.25          # grid length in beats (0.25 = 16th note)
    bpm: float = 120.0
    min_confidence: float = 0.3
    max_fret: int = 19
    bars_per_line: int = 4
    capo: int = 0
    difficulty: str = "standard"   # easy / standard / advanced
    chord_beats: int = 2           # preferred chord duration; strong changes may be shorter
    suppress_chord_attacks: bool = True
    lead_start_seconds: float = 0. # manual intro exclusion, 0 = automatic only
    audio_lead_detection: bool = True
    grid_offset: float = 0.0        # seconds; grid origin (beat 0) of the song
    rhythm_mode: str = "chords"    # chords = held accompaniment; notes = detected attacks


def notes_to_tab(notes: list[dict], opts: TabOptions | None = None, *, arranged: bool = False,
                 duration: float | None = None) -> str:
    """Render notes as ASCII tab text."""
    opts = opts or TabOptions()
    bpm = max(opts.bpm, 1.0)
    sec_per_beat = 60.0 / bpm
    grid = opts.quantize * sec_per_beat          # seconds per grid step
    steps_per_bar = int(round(4.0 / opts.quantize))
    steps_per_line = steps_per_bar * opts.bars_per_line

    planned = notes if arranged else arrange_notes(notes, opts)
    if not planned:
        return "(タブを生成できるノートが検出されませんでした)"
    cells: dict[int, list[tuple[int, int]]] = {}
    for n in planned:
        step = max(0, int(round((n["start"] - opts.grid_offset) / grid)))
        cells.setdefault(step, []).append((n["string"], n["fret"]))

    max_step = max(cells) if cells else 0
    if duration is not None:
        max_step = max(max_step, ceil((duration - opts.grid_offset) / grid) - 1)
    held = {}
    for n in planned:
        if n.get("chord"):
            first = max(0, int(round((n["start"] - opts.grid_offset) / grid)))
            last = max(first, ceil((n["end"] - opts.grid_offset) / grid) - 1)
            for step in range(first + 1, last + 1):
                held.setdefault(step, []).append(n["string"])
            max_step = max(max_step, last)
    total_lines = max_step // steps_per_line + 1

    tuning = opts.tuning
    n_strings = len(tuning)
    lines_out = []

    for line_idx in range(total_lines):
        start_step = line_idx * steps_per_line
        end_step = start_step + steps_per_line
        width = steps_per_line + steps_per_bar  # bar separators
        # each cell is 2 chars wide so two-digit frets (10-19) fit
        rows = [["--"] * width for _ in range(n_strings)]

        for step in range(start_step, min(end_step, max_step + 1)):
            offset = step - start_step
            col = offset + offset // steps_per_bar
            for string in held.get(step, []):
                rows[string][col] = "~~"

        for step in range(start_step, min(end_step, max_step + 1)):
            if step not in cells:
                continue
            offset = step - start_step
            col = offset + offset // steps_per_bar  # extra slot for each barline
            for s, f in cells[step]:
                rows[s][col] = str(f).rjust(2).replace(" ", "-")

        header = build_header(line_idx, opts)
        lines_out.append(header)
        for s in range(n_strings):
            label = f"{s + 1}|"
            lines_out.append(label + "".join(rows[s]) + "-|")
        lines_out.append("")

    return "\n".join(lines_out)


def build_header(line_idx: int, opts: TabOptions) -> str:
    start_bar = line_idx * opts.bars_per_line + 1
    return f"Bar {start_bar}-{start_bar + opts.bars_per_line - 1}"


def _candidates(pitch: int, opts: TabOptions) -> list[tuple[int, int]]:
    """All playable (string, fret) pairs for a MIDI pitch.
    Returns [] if unplayable. Strings are 0-based (0 = high E)."""
    out = []
    for s, open_pitch in enumerate(opts.tuning):
        fret = pitch - open_pitch - opts.capo
        if 0 <= fret <= opts.max_fret:
            out.append((s, fret))
    return out


# (maximum simultaneous notes, fret span, physical stretch in mm, barre allowed)
DIFFICULTIES = {
    "easy": (3, 3, 100.0, False),
    "standard": (4, 4, 125.0, True),
    "advanced": (6, 5, 150.0, True),
}


def playable(shape: tuple[tuple[int, int], ...], opts: TabOptions) -> bool:
    """Conservative four-finger model, with one index-finger barre at the nut side.

    Open strings do not contribute to hand stretch. A barre cannot cross an
    open string or a note fretted below it. This is a heuristic, not a guarantee
    for every player's hand or every guitar scale length.
    """
    limit, span, stretch, barre = DIFFICULTIES[opts.difficulty]
    if len(shape) > limit or len({s for s, _ in shape}) != len(shape):
        return False
    fretted = [(s, f) for s, f in shape if f > 0]
    if not fretted:
        return True
    lo, hi = min(f for _, f in fretted), max(f for _, f in fretted)
    # 648 mm scale; capo changes absolute fret spacing, not TAB fret numbers.
    distance = 648 * (2 ** (-(lo + opts.capo) / 12) - 2 ** (-(hi + opts.capo) / 12))
    if hi - lo > span or distance > stretch:
        return False
    fingers = len(fretted)
    if barre:
        strings = [s for s, f in fretted if f == lo]
        left, right = min(strings), max(strings)
        if not any(left <= s <= right and f < lo for s, f in shape):
            fingers -= len(strings) - 1
    return fingers <= 4


def _center(shape):
    frets = [f for _, f in shape if f > 0]
    return (min(frets) + max(frets)) / 2 if frets else None


def _effort(shape):
    frets = [f for _, f in shape if f > 0]
    return (0.08 * sum(frets) + 0.25 * (max(frets) - min(frets))) if frets else 0.0


def _chord_candidates(notes, opts):
    # Highest voice has priority, then bass, then confident inner voices.
    unique = {}
    for n in notes:
        if _candidates(n["pitch"], opts):
            if n["pitch"] not in unique or n["confidence"] > unique[n["pitch"]]["confidence"]:
                unique[n["pitch"]] = n
    pitches = sorted(unique, reverse=True)
    if not pitches:
        return [(0.0, (), ())]
    top, bass = pitches[0], pitches[-1]
    order = sorted(pitches, key=lambda p: (p != top, p != bass, -unique[p]["confidence"], -p))
    states = [(0.0, (), ())]
    for pitch in order:
        weight = (40 if pitch == top else 12 if pitch == bass else 4) + unique[pitch]["confidence"]
        expanded = []
        for penalty, shape, kept in states:
            expanded.append((penalty + weight, shape, kept))
            for pos in _candidates(pitch, opts):
                new_shape = shape + (pos,)
                if playable(new_shape, opts):
                    expanded.append((penalty, new_shape, kept + (unique[pitch],)))
        # Bound noisy/transcribed dense chords; retain multiple voicings for lookahead.
        states = sorted(expanded, key=lambda x: x[0] + _effort(x[1]))[:192]
    return sorted(states, key=lambda x: x[0] + _effort(x[1]))[:64]


def _transition(previous, current, seconds):
    a, b = _center(previous), _center(current)
    shift = abs(a - b) if a is not None and b is not None else 0
    strings = abs(sum(s for s, _ in previous) / max(1, len(previous))
                  - sum(s for s, _ in current) / max(1, len(current)))
    # A move in a fast passage is more costly than the same move after a rest.
    return (shift * 1.5 + strings * 0.25) / max(0.15, min(2.0, seconds))


def arrange_notes(notes: list[dict], opts: TabOptions | None = None) -> list[dict]:
    """Select playable voices and optimize voicings across all quantized onsets.

    Dynamic programming over bounded chord candidates, rather than choosing
    each chord greedily. Each onset releases the previous shape: this first
    version does not model independent sustained voices or legato techniques.
    Returned copies contain string/fret and quantized timing for every export.
    """
    opts = opts or TabOptions()
    if opts.difficulty not in DIFFICULTIES:
        raise ValueError("Unknown difficulty: " + opts.difficulty)
    if opts.quantize <= 0 or not opts.tuning or opts.bars_per_line < 1:
        raise ValueError("Invalid grid or tuning")
    grid = opts.quantize * 60 / max(opts.bpm, 1)
    groups = {}
    for n in notes:
        if n["confidence"] >= opts.min_confidence:
            step = max(0, int(round((n["start"] - opts.grid_offset) / grid)))
            groups.setdefault(step, []).append(n)
    steps = sorted(groups)
    if not steps:
        return []
    layers = []
    previous = []
    for index, step in enumerate(steps):
        layer = []
        for penalty, shape, kept in _chord_candidates(groups[step], opts):
            local = penalty + _effort(shape)
            if previous:
                seconds = (step - steps[index - 1]) * grid
                score, parent = min((node[0] + _transition(node[1], shape, seconds), j)
                                    for j, node in enumerate(previous))
            else:
                score, parent = 0.0, -1
            layer.append((score + local, shape, kept, parent))
        layers.append(layer)
        previous = layer
    chosen = min(range(len(previous)), key=lambda j: previous[j][0])
    path = []
    for layer in reversed(layers):
        node = layer[chosen]
        path.append(node)
        chosen = node[3]
    result = []
    for i, (step, node) in enumerate(zip(steps, reversed(path))):
        start = max(0.0, opts.grid_offset + step * grid)
        next_start = max(0.0, opts.grid_offset + steps[i + 1] * grid) if i + 1 < len(steps) else float("inf")
        for (string, fret), note in zip(node[1], node[2]):
            end = max(start + 0.01, note["end"])
            result.append(dict(note, string=string, fret=fret, start=start, end=min(end, next_start)))
    return result


def arrange_melody(notes: list[dict], opts: TabOptions) -> list[dict]:
    """Keep the lead monophonic even when nearby attacks snap to one grid cell.

    Round up at the manual entry boundary so quantization cannot move a note
    back into a user-specified silent intro.
    """
    grid = opts.quantize * 60 / max(opts.bpm, 1)
    cutoff = max(0, ceil((opts.lead_start_seconds - opts.grid_offset) / grid))
    cells = {}
    for n in notes:
        if n["start"] < opts.lead_start_seconds:
            continue
        step = max(cutoff, int(round((n["start"] - opts.grid_offset) / grid)))
        if step not in cells or (n["confidence"], n["end"] - n["start"]) > (cells[step]["confidence"], cells[step]["end"] - cells[step]["start"]):
            cells[step] = n
    quantized = [dict(n, start=opts.grid_offset + step * grid) for step, n in sorted(cells.items())]
    return arrange_notes(quantized, opts)


def assign_positions(pitches: list[int], prev_center: float,
                     opts: TabOptions) -> list[tuple[int, int]]:
    """Compatibility helper for a single chord; full TAB uses phrase planning."""
    candidates = _chord_candidates([dict(pitch=p, confidence=1.0) for p in pitches], opts)
    def cost(candidate):
        center = _center(candidate[1])
        return candidate[0] + _effort(candidate[1]) + (abs(center - prev_center) * 0.7 if center is not None else 0)
    return list(min(candidates, key=cost)[1])
