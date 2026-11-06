"""Convert transcribed notes into a guitar tablature."""
from __future__ import annotations

from dataclasses import dataclass, field

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


def notes_to_tab(notes: list[dict], opts: TabOptions | None = None) -> str:
    """Render notes as ASCII tab text."""
    opts = opts or TabOptions()
    bpm = max(opts.bpm, 1.0)
    sec_per_beat = 60.0 / bpm
    grid = opts.quantize * sec_per_beat          # seconds per grid step
    steps_per_bar = int(round(4.0 / opts.quantize))
    steps_per_line = steps_per_bar * opts.bars_per_line

    # filter + snap to grid
    valid = [n for n in notes if n["confidence"] >= opts.min_confidence]
    if not valid:
        return "(タブを生成できるノートが検出されませんでした)"

    cells: dict[int, list[tuple[int, int]]] = {}  # step -> [(string, fret), ...]
    for n in valid:
        step = int(round(n["start"] / grid))
        pos = pick_string_fret(n["pitch"], opts)
        if pos is None:
            continue
        cells.setdefault(step, []).append(pos)

    max_step = max(cells) if cells else 0
    total_lines = max_step // steps_per_line + 1

    tuning = opts.tuning
    n_strings = len(tuning)
    lines_out = []

    for line_idx in range(total_lines):
        start_step = line_idx * steps_per_line
        end_step = start_step + steps_per_line
        width = steps_per_line + steps_per_bar  # bar separators
        rows = [["-" for _ in range(width)] for _ in range(n_strings)]

        for step in range(start_step, min(end_step, max_step + 1)):
            if step not in cells:
                continue
            col = step - start_step + (step // steps_per_bar)  # gap for barline
            for s, f in cells[step]:
                label = str(f) if f < 10 else "abcdefgh"[f - 10]
                if rows[s][col] != "-":
                    rows[s][col] = label if f <= 9 else "X"  # collision: keep simple
                else:
                    rows[s][col] = label

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


def pick_string_fret(pitch: int, opts: TabOptions) -> tuple[int, int] | None:
    """Choose best (string, fret) for a MIDI pitch, favoring low frets and
    keeping pitch in range. Returns None if unplayable."""
    best = None
    best_score = None
    for s, open_pitch in enumerate(opts.tuning):
        fret = pitch - open_pitch - opts.capo
        if 0 <= fret <= opts.max_fret:
            # prefer low frets (ease of play)
            score = fret + (4 if fret > 12 else 0)
            if best_score is None or score < best_score:
                best, best_score = (s, fret), score
    return best
