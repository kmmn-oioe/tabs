"""PDF tablature from the exact arranged notes used for the on-screen TAB.

The JSON companion to each MusicXML file preserves fingering and timing; PDF
export never re-assigns notes to strings or converts standard notation to TAB.
"""
from __future__ import annotations

import json
from math import ceil
from pathlib import Path

from .tab import TabOptions


def data_path(sheet_path: str) -> Path:
    return Path(sheet_path).with_suffix('.tab.json')


def export_tab_data(notes, sheet_path, opts: TabOptions, title, changes, duration):
    data = {
        'version': 1, 'title': title, 'bpm': opts.bpm,
        'grid_offset': opts.grid_offset, 'quantize': opts.quantize,
        'tuning': opts.tuning, 'capo': opts.capo,
        'bars_per_line': opts.bars_per_line, 'duration': duration,
        'notes': [{k: n[k] for k in ('start', 'end', 'string', 'fret')}
                  | ({'chord': n['chord']} if n.get('chord') else {}) for n in notes],
        'chords': [{'start': c.start, 'end': c.end, 'name': c.name} for c in changes],
    }
    data_path(sheet_path).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def _layout(data):
    """Quantize with the same origin/grid as notes_to_tab, preserving fingering."""
    grid = data['quantize'] * 60 / max(data['bpm'], 1)
    origin = data['grid_offset']
    steps_per_bar = round(4 / data['quantize'])
    if grid <= 0 or steps_per_bar < 1:
        raise ValueError('Invalid TAB grid')
    cell = lambda t: max(0, round((t - origin) / grid))
    attacks, holds = {}, {}
    last = max(0, ceil((data['duration'] - origin) / grid) - 1)
    for n in data['notes']:
        step = cell(n['start'])
        attacks.setdefault(step, []).append((n['string'], n['fret']))
        last = max(last, step)
        if n.get('chord'):
            end = max(step, ceil((n['end'] - origin) / grid) - 1)
            for held in range(step + 1, end + 1):
                holds.setdefault(held, set()).add(n['string'])
            last = max(last, end)
    chords = {cell(c['start']): c['name'] for c in data['chords']}
    return steps_per_bar, last // steps_per_bar + 1, attacks, holds, chords


def tab_to_pdf(sheet_path: str, pdf_path: str):
    """Render six (or the configured number of) string lines and chord names."""
    source = data_path(sheet_path)
    if not source.is_file():
        raise RuntimeError('TAB譜のPDF用データがありません。最新版でもう一度タブ譜を生成してください。')
    data = json.loads(source.read_text(encoding='utf-8'))
    render_tab_pdf(data, pdf_path)


def render_tab_pdf(data, pdf_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen.canvas import Canvas

    steps, total_bars, attacks, holds, chords = _layout(data)
    page_w, page_h = A4
    margin, label_w = 38, 32
    available = page_w - 2 * margin - label_w
    string_gap = 12
    string_count = len(data['tuning'])
    row_height = (string_count - 1) * string_gap + 64
    bars_per_row = max(1, min(data['bars_per_line'], int(available / (steps*14 + 12))))
    labels = dict(chords)
    for b in range(0, total_bars, bars_per_row):
        if b * steps not in labels:
            at = data['grid_offset'] + b * steps * data['quantize'] * 60 / max(data['bpm'], 1)
            active = [ch for ch in data['chords'] if ch['start'] <= at < ch['end']]
            if active:
                labels[b * steps] = f"({active[-1]['name']})"
    # Reserve room for double-digit frets and adjacent chord changes.
    widths = [max(14., stringWidth(labels.get(i, ''), 'Helvetica-Bold', 10) + 5)
              for i in range(total_bars * steps)]
    bar_widths = [sum(widths[b*steps:(b+1)*steps]) + 12 for b in range(total_bars)]
    c = Canvas(str(pdf_path), pagesize=A4, pageCompression=1)
    c.setTitle(f"{data['title']} - Guitar TAB")
    names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    tuning = ' '.join(names[p % 12] for p in data['tuning'])
    page, y = 0, 0

    def new_page():
        nonlocal page, y
        if page:
            c.showPage()
        page += 1
        c.setFont('Helvetica-Bold', 15)
        c.drawString(margin, page_h - 38, f"{data['title']} - Guitar TAB")
        c.setFont('Helvetica', 9)
        c.drawString(margin, page_h - 56,
                     f"BPM {data['bpm']:g} | 4/4 | Capo {data['capo']} | Tuning (1-{string_count}): {tuning}")
        c.drawString(margin, page_h - 70, 'Frets relative to capo; chord names = sounding pitch (estimated).')
        c.drawString(margin, page_h - 83, '~ = hold chord; rest = no notes; numbers below TAB = beats.')
        c.drawRightString(page_w-margin, 22, f'Page {page}')
        y = page_h - 121

    new_page()
    bar = 0
    while bar < total_bars:
        if y - (string_count - 1) * string_gap - 25 < 40:
            new_page()
        end = min(total_bars, bar + bars_per_row)
        used = sum(bar_widths[bar:end])
        scale = min(1., available / used)
        x_start, x_end = margin + label_w, margin + label_w + used * scale
        c.setStrokeColorRGB(.35, .35, .35)
        c.setLineWidth(.5)
        for s in range(string_count):
            sy = y - s * string_gap
            c.line(x_start, sy, x_end, sy)
            c.setFont('Helvetica', 8)
            c.drawRightString(x_start - 8, sy - 3, f'{s+1}')
        c.setFont('Helvetica-Bold', 8)
        c.drawString(margin - 1, y - 28, 'TAB')
        x = x_start
        for b in range(bar, end):
            c.line(x, y + 3, x, y - (string_count-1)*string_gap - 3)
            c.setFont('Helvetica', 8)
            c.drawString(x + 2, y + 24, str(b+1))
            cx = x + 8 * scale
            for step in range(b*steps, (b+1)*steps):
                if step in labels:
                    c.setFont('Helvetica-Bold', 10*scale)
                    c.drawString(cx - 3, y + 11, labels[step])
                if step % steps * data['quantize'] % 1 < 1e-6:
                    c.setFont('Helvetica', 7)
                    c.drawCentredString(cx, y - (string_count-1)*string_gap - 17,
                                        str(int(step % steps * data['quantize']) + 1))
                notes = dict(attacks.get(step, []))
                for s in holds.get(step, set()):
                    if s not in notes:
                        c.setFillColorRGB(1, 1, 1)
                        c.rect(cx-4, y-s*string_gap-4, 8, 10, fill=1, stroke=0)
                        c.setFillColorRGB(0, 0, 0)
                        c.setFont('Helvetica', 8)
                        c.drawCentredString(cx, y - s*string_gap - 3, '~')
                for s, fret in notes.items():
                    text = str(fret)
                    c.setFont('Helvetica-Bold', 10*scale)
                    w = stringWidth(text, 'Helvetica-Bold', 10*scale) + 3
                    c.setFillColorRGB(1, 1, 1)
                    c.rect(cx-w/2, y-s*string_gap-4, w, 10, fill=1, stroke=0)
                    c.setFillColorRGB(0, 0, 0)
                    c.drawCentredString(cx, y-s*string_gap-3, text)
                cx += widths[step] * scale
            if not any(i in attacks or i in holds for i in range(b*steps, (b+1)*steps)):
                c.setFont('Helvetica-Oblique', 9)
                c.drawCentredString(x + bar_widths[b]*scale/2, y - 29, 'rest')
            x += bar_widths[b] * scale
        c.line(x, y+3, x, y-(string_count-1)*string_gap-3)
        y -= row_height
        bar = end
    c.save()
