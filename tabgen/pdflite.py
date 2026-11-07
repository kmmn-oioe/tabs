"""Render standard notation to PDF without MuseScore.

Parses the generated MusicXML with music21 and draws a treble-clef staff
using reportlab + an installed music font (pure Python).
Output is simplified: no beams/ties/dynamics, but correct pitches, rests,
chords and barlines.
"""
from __future__ import annotations

import os

def _diatonic(value) -> int:
    """Zero-based written staff position, including enharmonic spellings.

    Pitch classes alone cannot distinguish G-sharp from A-flat. In particular
    a natural-note-only lookup raises KeyError(8) for G-sharp/A-flat.
    """
    from music21.pitch import Pitch
    pitch = value if isinstance(value, Pitch) else Pitch(float(value))
    return pitch.diatonicNoteNum - 1


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
    "natural": 0x266E,
}


def _music_font(canvas):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [
        ("TabMusicSegoe", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "seguisym.ttf")),
        ("TabMusicNoto", "/usr/share/fonts/truetype/noto/NotoMusic-Regular.ttf"),
        ("TabMusicFreeSerif", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"),
    ]
    for name, path in candidates:
        if name in pdfmetrics.getRegisteredFontNames():
            return name
        if os.path.isfile(path):
            font = TTFont(name, path)
            if all(cp in font.face.charToGlyph for cp in GLYPHS.values()):
                pdfmetrics.registerFont(font)
                return name
    raise RuntimeError("楽譜用フォントが見つかりません。Segoe UI Symbol または Noto Music を用意するか、MuseScoreでPDF出力してください。")


def _note_y(diatonic, middle, half=3.5):
    return middle + (diatonic - B4_DIA) * half


def _accidental_text(pitch):
    accidental = pitch.accidental
    if accidental is None or accidental.displayStatus is False:
        return ""
    return {"sharp": "♯", "flat": "♭", "natural": "♮",
            "double-sharp": "♯♯", "double-flat": "♭♭"}.get(accidental.name, "")


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
    written = [_diatonic(p) for m in measures for el in m.notesAndRests
               if float(el.quarterLength) > 0 for p in getattr(el, "pitches", ())]
    above = max(staff_h/2+30, (max(written, default=B4_DIA)-B4_DIA)*half+30)
    below = max(staff_h/2+30, (B4_DIA-min(written, default=B4_DIA))*half+30)
    system_h = max(90, above+below+15)
    systems_per_page = max(1, int((page_h - margin_top - margin_bottom) // system_h))

    c = rlcanvas.Canvas(pdf_path, pagesize=A4)
    c.setTitle(title or "tabs")
    font = _music_font(c)

    def glyph(ch: str, x: float, y: float, size: float):
        c.setFont(font, size)
        c.drawString(x, y, ch)

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
            staff_mid = y_cursor - above
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
                    c.setFont("Helvetica", 12)
                    c.drawString(mx + 17, staff_mid + 2, "4")
                    c.drawString(mx + 17, staff_mid - 12, "4")

                for el in m.notesAndRests:
                    off = el.offset
                    left = 42 if b == 0 else 16
                    x = mx + left + float(off) / 4.0 * (measure_w - left - 12)
                    dur = float(el.quarterLength)
                    if dur <= 0:
                        continue  # Chord symbols/grace events are not extra sounding notes.
                    if el.isRest or not hasattr(el, "pitches") or not el.pitches:
                        g = ("rest_whole" if dur >= 4 else "rest_half" if dur >= 2
                             else "rest_quarter" if dur >= 1 else "rest_eighth" if dur >= 0.5
                             else "rest_16th")
                        glyph(chr(GLYPHS[g]), x, staff_mid - staff_h / 2 + 2, 18)
                        continue
                    pitches = sorted(el.pitches, key=lambda p: p.midi)
                    top = pitches[-1]
                    dia = _diatonic(top)
                    stem_up = dia < B4_DIA
                    stem_x = x + 4.4 if stem_up else x - 1.6
                    ys = [_note_y(_diatonic(p), staff_mid, half) for p in pitches]
                    stem_y0 = min(ys) if stem_up else max(ys)
                    stem_y1 = max(ys)+26 if stem_up else min(ys)-26
                    for p in pitches:
                        position = _diatonic(p)
                        py = _note_y(position, staff_mid, half)
                        accidental = _accidental_text(p)
                        if accidental:
                            glyph(accidental, x - 11 - 5*(len(accidental)-1), py - 4, 12)
                        direction = 1 if position > B4_DIA else -1
                        for step in range(6, abs(position-B4_DIA)+1, 2):
                            ledger_y = staff_mid + direction*step*half
                            c.line(x-7, ledger_y, x+7, ledger_y)
                        hollow = dur >= 2
                        c.setLineWidth(0.9)
                        if hollow:
                            c.setStrokeColorRGB(0, 0, 0)
                            c.ellipse(x - 4.5, py - 3.2, x + 4.5, py + 3.2, stroke=1, fill=0)
                        else:
                            c.ellipse(x - 4.5, py - 3.2, x + 4.5, py + 3.2, stroke=1, fill=1)
                    if 0.25 <= dur < 4:
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
