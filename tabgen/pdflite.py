"""Render standard notation to PDF without MuseScore.

Parses the generated MusicXML with music21 and draws a treble-clef staff
using reportlab + Windows' Segoe UI Symbol musical glyphs (pure Python).
Output is simplified: no beams/ties/dynamics, but correct pitches, rests,
chords and barlines.
"""
from __future__ import annotations

import os

STEPS = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}  # pitch class -> diatonic step


def _diatonic(midi: int) -> int:
    return (midi // 12) * 7 + STEPS[midi % 12]


# reference: B4 (middle staff line) = diatonic 34
B4_DIA = 34

# glyph codepoints in Segoe UI Symbol
GLYPHS = {
    "clef": 0x1D11E,        # 𝄞 treble
    "rest_whole": 0x1D13B,
    "rest_half": 0x1D13C,
    "rest_quarter": 0x1D13D,
    "rest_eighth": 0x1D13E,
    "rest_16th": 0x1D13F,
    "sharp": 0x266F,
    "flat": 0x266D,
}


def _music_font(canvas):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font = "SegoeUISymbol"
    if font not in pdfmetrics.getRegisteredFontNames():
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "seguisym.ttf")
        pdfmetrics.registerFont(TTFont(font, path))
    return font


def render_pdf(musicxml_path: str, pdf_path: str, title: str = ""):
    from music21 import converter
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rlcanvas

    score = converter.parse(musicxml_path)
    part = score.parts[0] if score.parts else score
    measures = list(part.getElementsByClass("Measure")) or [part]

    page_w, page_h = A4  # 595 x 842
    margin_x, margin_top, margin_bottom = 40, 70, 50
    line_gap = 7.0                      # staff line spacing (pt)
    staff_h = line_gap * 4
    half = line_gap / 2                 # distance per diatonic step
    measure_w = 130.0
    bars_per_line = max(1, int((page_w - 2 * margin_x) // measure_w))
    system_h = 62
    systems_per_page = int((page_h - margin_top - margin_bottom) // system_h)

    c = rlcanvas.Canvas(pdf_path, pagesize=A4)
    c.setTitle(title or "tabs")
    font = _music_font(c)

    def glyph(ch: str, x: float, y: float, size: float):
        c.setFont(font, size)
        c.drawString(x, y, ch)

    def note_y(dia: int, staff_base_y: float) -> float:
        # staff_base_y = y of middle line (B4)
        return staff_base_y + (B4_DIA - dia) * half

    total_bars = len(measures)
    bar_no = 0
    first_page = True
    while bar_no < total_bars:
        if not first_page:
            c.showPage()
        first_page = False

        # header on page 1
        y_cursor = page_h - margin_top
        sys_idx = 0
        while sys_idx < systems_per_page and bar_no < total_bars:
            staff_mid = y_cursor - staff_h / 2
            # staff lines
            c.setLineWidth(0.7)
            for i in range(5):
                y = staff_mid - staff_h / 2 + i * line_gap
                c.line(margin_x, y, margin_x + bars_per_line * measure_w, y)
            glyph(chr(GLYPHS["clef"]), margin_x + 2, staff_mid - 6, 30)

            for b in range(bars_per_line):
                if bar_no >= total_bars:
                    break
                m = measures[bar_no]
                mx = margin_x + b * measure_w
                # barline
                c.line(mx, staff_mid - staff_h / 2, mx, staff_mid + staff_h / 2)
                if b == 0:
                    glyph("4", mx + 14, staff_mid + 2, 12)
                    glyph("4", mx + 14, staff_mid - 14, 12)

                for el in m.notesAndRests:
                    off = el.offset
                    x = mx + 16 + float(off) / 4.0 * (measure_w - 30)
                    dur = float(el.quarterLength)
                    if el.isRest or not hasattr(el, "pitches") or not el.pitches:
                        g = ("rest_whole" if dur >= 4 else "rest_half" if dur >= 2
                             else "rest_quarter" if dur >= 1 else "rest_eighth" if dur >= 0.5
                             else "rest_16th")
                        glyph(chr(GLYPHS[g]), x, staff_mid - staff_h / 2 + 2, 18)
                        continue
                    pitches = sorted(el.pitches, key=lambda p: p.midi)
                    top = pitches[-1]
                    dia = _diatonic(top.midi)
                    stem_up = dia < B4_DIA
                    stem_x = x + 4.4 if stem_up else x - 1.6
                    stem_y0 = note_y(dia, staff_mid)
                    stem_y1 = stem_y0 + 26 if stem_up else stem_y0 - 26
                    # accidental of top note
                    pc = top.midi % 12
                    if pc in (1, 3, 6, 8, 10):
                        glyph(chr(GLYPHS["sharp"]), x - 9, stem_y0 - 4, 12)
                    elif pc in (5, 2, 7, 10 - 1):
                        pass
                    for p in pitches:
                        py = note_y(_diatonic(p.midi), staff_mid)
                        hollow = dur >= 2
                        c.setLineWidth(0.9)
                        if hollow:
                            c.setStrokeColorRGB(0, 0, 0)
                            c.ellipse(x - 4.5, py - 3.2, x + 4.5, py + 3.2, stroke=1, fill=0)
                        else:
                            c.ellipse(x - 4.5, py - 3.2, x + 4.5, py + 3.2, stroke=1, fill=1)
                    if dur >= 0.25:
                        c.line(stem_x, stem_y0, stem_x, stem_y1)
                    if dur <= 0.5 and dur >= 0.49:
                        # eighth-note flag (simple curve)
                        fy = stem_y1 - 1 if stem_up else stem_y1 + 1
                        c.bezier(stem_x, fy, stem_x + 7, fy + 4, stem_x + 9, fy + 9, stem_x + 5, fy - 4 if not stem_up else fy + 12)
                bar_no += 1
            # final barline of the line
            c.line(margin_x + bars_per_line * measure_w,
                   staff_mid - staff_h / 2,
                   margin_x + bars_per_line * measure_w,
                   staff_mid + staff_h / 2)
            y_cursor -= system_h
            sys_idx += 1

        # page footer
        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w / 2, 25, f"{title}  -  page {c.getPageNumber()}")
    c.save()
