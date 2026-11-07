import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reportlab.pdfgen.canvas import Canvas

from tabgen.tab import TabOptions, arrange_notes
from tabgen.rhythm import ChordChange
from tabgen.tabpdf import export_tab_data, data_path, _layout, tab_to_pdf, render_tab_pdf


class RecordingCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = []
        self.pages = 1

    def drawString(self, x, y, text, *args, **kwargs):
        self.labels.append((x, y, text))
        return super().drawString(x, y, text, *args, **kwargs)

    def drawCentredString(self, x, y, text, *args, **kwargs):
        self.labels.append((x, y, text))
        return super().drawCentredString(x, y, text, *args, **kwargs)

    def showPage(self):
        self.pages += 1
        return super().showPage()


class TabPdfTests(unittest.TestCase):
    def make_data(self, tmp, duration=10):
        path = str(Path(tmp)/'rhythm.musicxml')
        opts = TabOptions(bpm=120, grid_offset=.25, capo=2,
                          tuning=[62, 57, 53, 48, 43, 38])
        notes = [dict(start=.25, end=3.25, string=0, fret=12, chord='D'),
                 dict(start=.25, end=3.25, string=5, fret=0, chord='D'),
                 dict(start=3.25, end=4.25, string=2, fret=10, chord='G#m7')]
        changes = [ChordChange(.25, 3.25, 2, '', (0,4,7)),
                   ChordChange(3.25, 4.25, 8, 'm7', (0,3,7,10))]
        export_tab_data(notes, path, opts, 'rhythm', changes, duration)
        return path, json.loads(data_path(path).read_text())

    def test_exact_fingering_offset_capo_and_chord_timing_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self.make_data(tmp)
        steps, bars, attacks, holds, symbols = _layout(data)
        self.assertEqual(steps, 16)
        self.assertEqual(bars, 5)
        self.assertEqual(attacks[0], [(0,12),(5,0)])
        self.assertEqual(attacks[24], [(2,10)])
        self.assertEqual(holds[16], {0,5})
        self.assertEqual(symbols, {0:'D',24:'G#m7'})
        self.assertEqual(data['capo'], 2)
        self.assertEqual(data['tuning'][-1], 38)

    def test_pdf_has_frets_on_strings_and_chords_above(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, data = self.make_data(tmp)
            data['bars_per_line'] = 1
            data_path(source).write_text(json.dumps(data))
            target = Path(tmp)/'tab.pdf'
            c = RecordingCanvas(str(target))
            with patch('reportlab.pdfgen.canvas.Canvas', return_value=c):
                tab_to_pdf(source, str(target))
            self.assertTrue(target.read_bytes().startswith(b'%PDF-'))
            labels = c.labels
        high = next((x,y) for x,y,t in labels if t == '12')
        low = next((x,y) for x,y,t in labels if t == '0')
        chord = next((x,y) for x,y,t in labels if t == 'D')
        self.assertEqual(high[0], low[0])
        self.assertEqual(high[1] - low[1], 60)
        self.assertGreater(chord[1], high[1])
        self.assertTrue(any(t == 'G#m7' for _,_,t in labels))
        self.assertTrue(any('Capo 2' in t for _,_,t in labels))
        self.assertTrue(any(t == '(D)' for _,_,t in labels))
        self.assertTrue(any(t == 'rest' for _,_,t in labels))

    def test_silent_lead_has_no_fabricated_frets_and_paginates(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self.make_data(tmp, duration=90)
            data['notes'] = []
            c = RecordingCanvas(str(Path(tmp)/'silent.pdf'))
            with patch('reportlab.pdfgen.canvas.Canvas', return_value=c):
                render_tab_pdf(data, str(Path(tmp)/'silent.pdf'))
            self.assertGreater(c.pages, 2)
            self.assertEqual(_layout(data)[2], {})
            self.assertTrue(any(t == 'rest' for _,_,t in c.labels))

    def test_missing_generation_data_requests_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'もう一度タブ譜を生成'):
                tab_to_pdf(str(Path(tmp)/'old.musicxml'), str(Path(tmp)/'tab.pdf'))

    def test_pipeline_exports_exact_arrangement_and_shared_harmony(self):
        from tabgen.pipeline import generate
        raw = [dict(pitch=p, start=0., end=2., confidence=.9) for p in [48,52,55]]
        for split in [False, True]:
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp)/'audio.wav'
                audio.touch()
                opts = TabOptions(audio_lead_detection=False)
                with patch('tabgen.transcribe.transcribe', return_value=raw), \
                     patch('tabgen.harmony.audio_chord_changes', return_value=[ChordChange(0,2,0,'',(0,4,7))]):
                    _, _, sheets, _ = generate(str(audio), opts, split_lead_rhythm=split, auto_bpm=False)
                data = [json.loads(data_path(p).read_text()) for p in sheets.split(';')]
                if split:
                    self.assertEqual(data[0]['chords'], data[1]['chords'])
                    self.assertEqual(data[1]['chords'][0]['name'], 'C')
                    self.assertEqual(data[0]['notes'], [])
                else:
                    expected = arrange_notes(raw, opts)
                    self.assertEqual(data[0]['notes'], [{k:n[k] for k in ('start','end','string','fret')} for n in expected])
                    self.assertEqual(data[0]['chords'][0]['name'], 'C')
                for p in sheets.split(';'):
                    tab_to_pdf(p, str(Path(tmp)/(Path(p).stem+'.pdf')))


if __name__ == '__main__':
    unittest.main()
