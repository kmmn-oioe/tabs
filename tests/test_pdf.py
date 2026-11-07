import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from music21 import chord, note, pitch, stream

from tabgen.pdflite import _diatonic, _note_y, _accidental_text, _music_font, render_pdf


def write_score(path, long=False):
    part = stream.Part()
    for name in ['C4','C#4','D4','E-4','E4','F4','F#4','G4','G#4','A4','B-4','B4']:
        part.append(note.Note(name,quarterLength=.5))
    for _ in range(36 if long else 1):
        part.append(chord.Chord(['E2','G#2','B2','D-4','E4'],quarterLength=4))
        part.append(note.Rest(quarterLength=.5))
    score = stream.Score()
    score.insert(0,part)
    score.write('musicxml',fp=path)


class PdfTests(unittest.TestCase):
    def test_all_midi_pitch_classes_work_including_eight(self):
        for midi in range(128):
            self.assertIsInstance(_diatonic(midi),int)
        self.assertEqual(_diatonic(68),32)  # Previously KeyError: 8.

    def test_staff_positions_use_written_spelling_and_correct_octave_direction(self):
        self.assertEqual(_diatonic(pitch.Pitch('B4')),34)
        self.assertEqual(_diatonic(pitch.Pitch('C4')),28)
        self.assertEqual(_diatonic(pitch.Pitch('G#4')),32)
        self.assertEqual(_diatonic(pitch.Pitch('A-4')),33)
        self.assertLess(_note_y(_diatonic(pitch.Pitch('C4')),100),100)
        self.assertGreater(_note_y(_diatonic(pitch.Pitch('C5')),100),100)

    def test_accidentals_cover_chord_members_flats_and_cancellations(self):
        self.assertEqual(_accidental_text(pitch.Pitch('G#4')),'♯')
        self.assertEqual(_accidental_text(pitch.Pitch('A-4')),'♭')
        p = pitch.Pitch('C4')
        p.accidental = pitch.Accidental('natural')
        self.assertEqual(_accidental_text(p),'♮')
        p.accidental.displayStatus = False
        self.assertEqual(_accidental_text(p),'')

    def test_render_chromatic_notes_and_chords_to_real_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml,pdf = Path(tmp)/'chromatic.musicxml',Path(tmp)/'chromatic.pdf'
            write_score(xml)
            render_pdf(str(xml),str(pdf),'Chromatic test')
            data = pdf.read_bytes()
            self.assertTrue(data.startswith(b'%PDF-'))
            self.assertTrue(data.rstrip().endswith(b'%%EOF'))
            self.assertGreater(len(data),1000)

    def test_low_guitar_notes_and_many_measures_paginate(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml,pdf = Path(tmp)/'long.musicxml',Path(tmp)/'long.pdf'
            write_score(xml,long=True)
            render_pdf(str(xml),str(pdf))
            self.assertGreater(len(re.findall(rb'/Type\s*/Page\b',pdf.read_bytes())),1)

    def test_missing_font_reports_actionable_error(self):
        with patch('reportlab.pdfbase.pdfmetrics.getRegisteredFontNames',return_value=[]), \
             patch('tabgen.pdflite.os.path.isfile',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'楽譜用フォント'):
                _music_font(None)

    def test_app_save_uses_real_fallback_renderer_without_musescore(self):
        from main import App
        with tempfile.TemporaryDirectory() as tmp:
            xml,pdf = Path(tmp)/'chromatic.musicxml',Path(tmp)/'result.pdf'
            write_score(xml)
            app = SimpleNamespace(sheet_paths=[str(xml)],status=Mock(),update=Mock())
            with patch('tabgen.pdf.find_musescore',return_value=None), \
                 patch('main.filedialog.asksaveasfilename',return_value=str(pdf)), \
                 patch('main.messagebox.showerror') as error:
                App._save_pdf(app)
            error.assert_not_called()
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))

    def test_app_error_includes_exception_type_not_just_a_number(self):
        from main import App
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp)/'score.musicxml'
            xml.touch()
            app = SimpleNamespace(sheet_paths=[str(xml)],status=Mock(),update=Mock())
            with patch('tabgen.pdf.find_musescore',return_value=None), \
                 patch('main.filedialog.asksaveasfilename',return_value=str(Path(tmp)/'out.pdf')), \
                 patch('tabgen.pdflite.render_pdf',side_effect=KeyError(8)), \
                 patch('main.messagebox.showerror') as error:
                App._save_pdf(app)
            self.assertIn('KeyError: 8',error.call_args.args[1])
            app.status.config.assert_any_call(text='PDF保存に失敗しました')


if __name__ == '__main__':
    unittest.main()
