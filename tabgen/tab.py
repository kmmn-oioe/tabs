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
    grid_offset: float = 0.0        # seconds; grid origin (beat 0) of the song


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
    prev_center = 7.0  # running hand-position center (fret), used for continuity
    for step in sorted({max(0, int(round((n["start"] - opts.grid_offset) / grid)))
                        for n in valid}):
        pitches = [n["pitch"] for n in valid
                   if max(0, int(round((n["start"] - opts.grid_offset) / grid))) == step]
        positions = assign_positions(pitches, prev_center, opts)
        if positions:
            cells[step] = positions
            frets = [f for _, f in positions]
            prev_center = (min(frets) + max(frets)) / 2.0

    max_step = max(cells) if cells else 0
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


def assign_positions(pitches: list[int], prev_center: float,
                     opts: TabOptions) -> list[tuple[int, int]]:
    """Assign (string, fret) for all pitches sounding at one grid step.

    Chord-level optimization: no string may carry two pitches, the used
    fret span is minimized, pitches are kept on strings natural for their
    register, and the hand position stays close to the previous step.
    Pitches the instrument cannot voice are dropped.
    """
    if not pitches:
        return []
    # deduplicate while keeping pitch order (low -> high)
    uniq = sorted(set(pitches))
    cand = [_candidates(p, opts) for p in uniq]
    keep = [i for i, c in enumerate(cand) if c]
    uniq = [uniq[i] for i in keep]
    cand = [cand[i] for i in keep]
    if not cand:
        return []

    # natural string per pitch: highest string whose open pitch <= pitch
    def natural_string(p: int) -> int:
        best = 0
        for s, op in enumerate(opts.tuning):
            if p >= op - opts.capo:
                best = s
        return best

    if len(uniq) == 1:
        combos = [(c, ) for c in cand[0]]
    else:
        total = 1
        for c in cand:
            total *= len(c)
        if total <= 20000:  # brute-force small chords
            import itertools
            combos = itertools.product(*cand)
        else:  # huge chord: greedy, high pitch on high string first
            order = sorted(range(len(uniq)), key=lambda i: -uniq[i])
            chosen: dict[int, tuple[int, int]] = {}
            used: set[int] = set()
            for i in order:
                best_c, best_sc = None, None
                for sc in cand[i]:
                    if sc[0] in used:
                        continue
                    sc_score = abs(sc[0] - natural_string(uniq[i])) * 2 + sc[1] * 0.2
                    if best_sc is None or sc_score < best_sc:
                        best_c, best_sc = sc, sc_score
                if best_c:
                    chosen[i] = best_c
                    used.add(best_c[0])
            combos = [tuple(chosen.get(i, cand[i][0]) for i in range(len(uniq)))]

    def score(combo) -> float:
        frets = [f for _, f in combo]
        strings = [s for s, _ in combo]
        total = sum(frets) * 0.2
        total += max(frets) - min(frets)              # compact hand shape
        total += sum(abs(s - natural_string(p)) * 1.5     # natural register
                     for s, p in zip(strings, uniq))
        center = (min(frets) + max(frets)) / 2.0
        total += abs(center - prev_center) * 0.7      # position continuity
        return total

    best_combo, best_score = None, None
    for combo in combos:
        sc = score(combo)
        if best_score is None or sc < best_score:
            best_combo, best_score = combo, sc
    return list(best_combo) if best_combo else []
