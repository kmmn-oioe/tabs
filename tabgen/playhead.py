"""Map generated ASCII TAB cells to audio time and draw a non-editing cursor."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
import re
import tkinter as tk


@dataclass(frozen=True)
class PlaybackTiming:
    bpm: float
    quantize: float
    offset: float
    strings: int = 6

    @classmethod
    def from_options(cls, opts):
        return cls(max(1., opts.bpm), opts.quantize, opts.grid_offset, len(opts.tuning))

    @property
    def steps_per_bar(self):
        return int(round(4 / self.quantize))

    @property
    def step_seconds(self):
        return self.quantize * 60 / self.bpm


@dataclass(frozen=True)
class TabBlock:
    part: str
    first_bar: int
    last_bar: int
    rows: tuple[int, ...]   # Text widget line numbers, 1-based


@dataclass(frozen=True)
class CursorCell:
    block: TabBlock
    column: float
    cell_column: int


def parse_blocks(content: str, timing: PlaybackTiming) -> list[TabBlock]:
    if (not all(isfinite(x) for x in (timing.bpm, timing.quantize, timing.offset))
            or timing.bpm <= 0 or timing.quantize <= 0 or timing.steps_per_bar < 1):
        return []
    lines, blocks, part = content.splitlines(), [], "タブ譜"
    for i, line in enumerate(lines):
        heading = re.fullmatch(r"=+\s*(.*?)\s*=+", line)
        if heading:
            part = heading[1].strip()
        header = re.fullmatch(r"Bar (\d+)-(\d+)", line)
        if not header:
            continue
        first, last = map(int, header.groups())
        if first < 1 or last < first:
            continue
        # The renderer reserves one extra bar's worth of padding at line end.
        body_length = 2 * timing.steps_per_bar * (last-first+2) + 1
        rows = []
        for string in range(1, timing.strings+1):
            index = i+string
            if index >= len(lines):
                break
            row = re.fullmatch(rf"{string}\|([0-9~\-]+)\|", lines[index])
            if row is None or len(row[1]) != body_length:
                break
            rows.append(index+1)
        if len(rows) == timing.strings:
            blocks.append(TabBlock(part, first, last, tuple(rows)))
    return blocks


def cursor_cells(blocks, timing: PlaybackTiming, seconds: float) -> list[CursorCell]:
    if not isfinite(seconds) or seconds < timing.offset:
        return []
    step = (seconds - timing.offset) / timing.step_seconds
    whole = floor(step + 1e-9)
    fraction = max(0., step - whole)
    result = []
    for block in blocks:
        first = (block.first_bar-1)*timing.steps_per_bar
        if not first <= whole < block.last_bar*timing.steps_per_bar:
            continue
        offset = whole-first
        # Match notes_to_tab's two-character frets and extra bar separator cells.
        a = 2 + 2*(offset + offset//timing.steps_per_bar)
        b = 2 + 2*(offset+1 + (offset+1)//timing.steps_per_bar)
        result.append(CursorCell(block, a+(b-a)*fraction, a))
    return result


class TabPlayhead:
    """Place thin overlays over visible string rows without altering TAB text."""
    def __init__(self, text, on_parts=None):
        self.text = text
        self.on_parts = on_parts
        self.timing = None
        self.blocks = []
        self.part = "タブ譜"
        self.seconds = 0.
        self._bars = []
        self._job = None
        text.tag_configure("playhead_cell", background="#fff0bd")
        text.bind("<<Modified>>", self._modified, add="+")
        text.bind("<Configure>", lambda event: self.schedule_draw(), add="+")

    def set_timing(self, timing):
        self.timing = timing
        self.refresh()

    def refresh(self):
        self.blocks = parse_blocks(self.text.get("1.0", "end-1c"), self.timing) if self.timing else []
        parts = list(dict.fromkeys(b.part for b in self.blocks))
        if self.part not in parts:
            self.part = parts[0] if parts else "タブ譜"
        if self.on_parts:
            self.on_parts(parts, self.part)
        self.schedule_draw()

    def _modified(self, event):
        if self.text.edit_modified():
            self.text.edit_modified(False)
            self.refresh()

    def schedule_draw(self):
        if self._job is None:
            self._job = self.text.after_idle(self._draw_idle)

    def _draw_idle(self):
        self._job = None
        self.update(self.seconds)

    def select_part(self, part):
        self.part = part
        self.update(self.seconds, follow=True)

    def update(self, seconds, *, follow=False):
        self.seconds = seconds
        cells = cursor_cells(self.blocks, self.timing, seconds) if self.timing else []
        selected = next((c for c in cells if c.block.part == self.part), None)
        if follow and selected:
            for row in (selected.block.rows[0], selected.block.rows[-1]):
                self.text.see(f"{row}.{floor(selected.column)}")
        self.text.tag_remove("playhead_cell", "1.0", "end")
        for bar in self._bars:
            bar.place_forget()
        while len(self._bars) < len(cells):
            self._bars.append(tk.Frame(self.text, background="#e53935", width=2, borderwidth=0,
                                       takefocus=False))
        for bar, cell in zip(self._bars, cells):
            visible = []
            column = floor(cell.column)
            for row in cell.block.rows:
                self.text.tag_add("playhead_cell", f"{row}.{cell.cell_column}", f"{row}.{cell.cell_column+2}")
                line = self.text.dlineinfo(f"{row}.0")
                box = self.text.bbox(f"{row}.{column}")
                if line and box:
                    visible.append((line, box))
            if visible:
                top = min(line[1] for line, _ in visible)
                bottom = max(line[1]+line[3] for line, _ in visible)
                box = visible[0][1]
                x = box[0]+(cell.column-column)*box[2]
                bar.place(x=round(x), y=top, width=2, height=bottom-top, bordermode="ignore")
                bar.lift()
        self.text.tag_raise("sel")

    def close(self):
        if self._job is not None:
            self.text.after_cancel(self._job)
            self._job = None
        for bar in self._bars:
            bar.destroy()
        self._bars.clear()
