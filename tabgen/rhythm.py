"""Beat-aligned chord accompaniment inferred from duration-weighted pitch classes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import ceil, floor

from .tab import TabOptions, DIFFICULTIES, playable, _effort, _transition

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITIES = (
    ("", (0, 4, 7)), ("m", (0, 3, 7)),
    ("7", (0, 4, 7, 10)), ("maj7", (0, 4, 7, 11)),
    ("m7", (0, 3, 7, 10)), ("dim", (0, 3, 6)),
    ("sus2", (0, 2, 7)), ("sus4", (0, 5, 7)), ("5", (0, 7)),
)


@dataclass(frozen=True)
class ChordChange:
    start: float
    end: float
    root: int | None = None
    suffix: str = ""
    intervals: tuple[int, ...] = ()

    @property
    def name(self):
        return "N.C." if self.root is None else NAMES[self.root] + self.suffix


def _rank_chords(weights, bass_weights):
    if not weights:
        return []
    # Ignore very weak short ornaments rather than treating them as extensions.
    weights = {pc: w for pc, w in weights.items() if w >= max(weights.values()) * .12}
    total = sum(weights.values())
    if total <= 0:
        return []
    ranked = []
    for root in weights:
        for suffix, intervals in QUALITIES:
            pcs = {(root + interval) % 12 for interval in intervals}
            heard = pcs & weights.keys()
            if len(heard) < 2 or (len(intervals) > 3 and len(heard) < 3):
                continue
            # Do not invent a major/minor third from just a root and fifth.
            if len(intervals) >= 3 and not any((root + i) % 12 in weights for i in intervals[1:] if i != 7):
                continue
            inside = sum(weights.get(pc, 0) for pc in pcs) / total
            missing = len(pcs - weights.keys()) / len(pcs)
            bass = bass_weights.get(root, 0) / max(sum(bass_weights.values()), 1e-9)
            score = inside - 1.2 * (1 - inside) - .8 * missing + .08 * bass - .025 * len(intervals)
            ranked.append((score, root, suffix, intervals))
    return sorted(ranked, key=lambda candidate: candidate[0], reverse=True)


def _identify(weights, bass_weights):
    ranked = _rank_chords(weights, bass_weights)
    best = ranked[0] if ranked else None
    # A poorly explained/noisy window is left unharmonized.
    return best[1:] if best is not None and best[0] >= .35 else (None, "", ())


def chord_changes(notes: list[dict], opts: TabOptions) -> list[ChordChange]:
    if opts.chord_beats not in (1, 2, 4):
        raise ValueError("Chord interval must be 1, 2 or 4 beats")
    valid = [n for n in notes if n["confidence"] >= opts.min_confidence and n["end"] > max(0, n["start"])]
    if not valid:
        return []
    duration = opts.chord_beats * 60 / max(opts.bpm, 1)
    origin = opts.grid_offset
    end = max(n["end"] for n in valid)
    count = max(1, ceil((end - origin) / duration))
    windows = [[] for _ in range(count)]
    for n in valid:
        first = max(0, floor((n["start"] - origin) / duration))
        last = min(count, max(1, ceil((n["end"] - origin) / duration)))
        for i in range(first, last):
            windows[i].append(n)
    changes = []
    for i, window in enumerate(windows):
        start = max(0., origin + i * duration)
        if i == 0:
            start = min(start, max(0., min(n["start"] for n in valid)))
        stop = min(end, origin + (i + 1) * duration)
        if stop <= start:
            continue
        weights, bass_weights = {}, {}
        lowest = min((n["pitch"] for n in window), default=0)
        for n in window:
            overlap = max(0., min(stop, n["end"]) - max(start, n["start"]))
            w = overlap * n["confidence"]
            pc = n["pitch"] % 12
            weights[pc] = weights.get(pc, 0) + w
            if n["pitch"] == lowest:
                bass_weights[pc] = bass_weights.get(pc, 0) + w
        root, suffix, intervals = _identify(weights, bass_weights)
        change = ChordChange(start, stop, root, suffix, intervals)
        if changes and changes[-1].name == change.name:
            changes[-1] = replace(changes[-1], end=stop)
        else:
            changes.append(change)
    return changes


def _voicing_cost(shape, change, opts):
    if not shape:
        return 0.
    bass = min(opts.tuning[s] + opts.capo + f for s, f in shape) % 12
    strings = [s for s, _ in shape]
    skipped = max(strings) - min(strings) + 1 - len(strings)
    # Prefer a root bass and adjacent strings that can be strummed together.
    return _effort(shape) + 1.2 * (bass != change.root) + skipped - .35 * len(shape)


def _voicings(change, opts):
    if change.root is None:
        return [()]
    limit, span, _, _ = DIFFICULTIES[opts.difficulty]
    intervals = change.intervals
    # A three-note seventh voicing retains root, third and seventh, omitting fifth.
    if len(intervals) > limit:
        intervals = tuple(i for i in intervals if i != 7)
    pcs = {(change.root + i) % 12 for i in intervals}
    shapes = set()
    for anchor in range(opts.max_fret + 1):
        choices = []
        for s, tuning in enumerate(opts.tuning):
            frets = {0, *range(max(1, anchor), min(opts.max_fret, anchor + span) + 1)}
            choices.append([None] + [(s, f) for f in sorted(frets) if (tuning + opts.capo + f) % 12 in pcs])
        for combo in product(*choices):
            shape = tuple(pos for pos in combo if pos is not None)
            if len(shape) > limit or len(shape) < len(pcs):
                continue
            if {(opts.tuning[s] + opts.capo + f) % 12 for s, f in shape} != pcs:
                continue
            if playable(shape, opts):
                shapes.add(shape)

    return sorted(shapes, key=lambda shape: (_voicing_cost(shape, change, opts), shape))[:64]


def arrange_rhythm(notes: list[dict], opts: TabOptions, *,
                   changes: list[ChordChange] | None = None) -> tuple[list[dict], list[ChordChange]]:
    """Hold one playable voicing until the chord changes; silence remains silence.

    Uses the full transcription, including arpeggios, to retain chord evidence
    that would otherwise disappear when the top note is split into the lead.
    """
    changes = chord_changes(notes, opts) if changes is None else list(changes)
    if not changes:
        return [], []
    cache, layers = {}, []
    for i, change in enumerate(changes):
        key = change.root, change.intervals
        if key not in cache:
            cache[key] = _voicings(change, opts)
        shapes = cache[key]
        if not shapes:
            change = replace(change, root=None, suffix="", intervals=())
            changes[i] = change
            shapes = [()]
        layer = []
        for shape in shapes:
            local = _voicing_cost(shape, change, opts)
            if i:
                elapsed = change.start - changes[i - 1].start
                score, parent = min((node[0] + _transition(node[1], shape, elapsed), j)
                                    for j, node in enumerate(layers[-1]))
            else:
                score, parent = 0., -1
            layer.append((score + local, shape, parent))
        layers.append(layer)
    chosen = min(range(len(layers[-1])), key=lambda j: layers[-1][j][0])
    path = []
    for layer in reversed(layers):
        node = layer[chosen]
        path.append(node[1])
        chosen = node[2]
    result = []
    for change, shape in zip(changes, reversed(path)):
        for s, f in shape:
            result.append(dict(start=change.start, end=change.end,
                               pitch=opts.tuning[s] + opts.capo + f, confidence=1.,
                               string=s, fret=f, chord=change.name))
    return result, changes


def progression_text(changes: list[ChordChange], opts: TabOptions) -> str:
    lines = ["コード進行（実音・~ は次の切替まで保持）"]
    beat = 60 / max(opts.bpm, 1)
    for c in changes:
        position = max(0., (c.start - opts.grid_offset) / beat)
        bar, within = int(position // 4) + 1, position % 4 + 1
        lines.append(f"{bar}小節 {within:g}拍: {c.name}  [{c.start:.2f}–{c.end:.2f}秒]")
    return "\n".join(lines)
