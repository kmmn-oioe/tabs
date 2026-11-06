import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from tabgen.split import split_notes
from tabgen.tab import TabOptions, arrange_melody


def note(pitch, start, end=None):
    return dict(pitch=pitch, start=start, end=start+.5 if end is None else end, confidence=.9)


def intro():
    # Eleven seconds of overlapping, staggered chord attacks followed by a line.
    return [note(p, i+j*.02, i+.7) for i in range(11)
            for j,p in enumerate([45, 52, 57, 60])]


class LeadPresenceTests(unittest.TestCase):
    def test_rhythm_only_intro_is_not_a_lead(self):
        notes = intro() + [note(64, 11), note(62, 11.3), note(60, 11.6)]
        lead, rhythm = split_notes(notes)
        self.assertEqual([n['pitch'] for n in lead], [64, 62, 60])
        self.assertEqual(len(rhythm), len(intro()))

    def test_entire_chord_track_can_have_no_lead(self):
        lead, rhythm = split_notes(intro())
        self.assertEqual(lead, [])
        self.assertEqual(len(rhythm), len(intro()))

    def test_fast_nonoverlapping_melody_is_not_a_strum(self):
        notes = [note(p, i*.025, i*.025+.02) for i,p in enumerate([60,62,64,65,67])]
        lead, _ = split_notes(notes, window=.01)
        self.assertEqual([n['pitch'] for n in lead], [60,62,64,65,67])

    def test_manual_entry_removes_ambiguous_arpeggio_but_keeps_later_notes(self):
        notes = [note([57,60,64][i % 3], i*.25) for i in range(44)] + [note(66, 11.01), note(64, 11.4)]
        lead, _ = split_notes(notes, start_seconds=11, suppress_chords=False)
        self.assertEqual([n['pitch'] for n in lead], [66,64])
        opts = TabOptions(lead_start_seconds=11, bpm=107.7, grid_offset=.14, audio_lead_detection=False)
        arranged = arrange_melody(lead, opts)
        self.assertTrue(all(n['start'] >= 11 for n in arranged))

    def test_grid_does_not_turn_fast_melody_into_chord(self):
        opts = TabOptions()
        notes = [note(60, .01), note(62, .05), note(64, .14)]
        arranged = arrange_melody(notes, opts)
        self.assertEqual(len(arranged), 2)
        self.assertEqual(len(arranged), len({n['start'] for n in arranged}))

    def test_invalid_entry_is_rejected(self):
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                split_notes([], start_seconds=value)

    def test_eleven_second_rest_in_all_lead_exports(self):
        from tabgen.pipeline import generate
        import pretty_midi
        from music21 import converter
        opts = TabOptions(lead_start_seconds=11, bpm=107.7, grid_offset=.14, audio_lead_detection=False)
        notes = [note(57, i*.25) for i in range(44)] + [note(66, 11.01), note(64, 11.4)]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'audio.wav'
            source.touch()
            with patch('tabgen.transcribe.transcribe', return_value=notes), \
                 patch('tabgen.harmony.audio_chord_changes', return_value=[]):
                text, midis, xmls, _ = generate(str(source), opts, split_lead_rhythm=True, auto_bpm=False)
            lead_text = text.split('====== リズムギター')[0]
            self.assertIn('リード開始指定: 11秒', lead_text)
            # Four bars (16 beats) are completely silent before the specified entry.
            first_block = lead_text.split('Bar 1-4')[1].split('Bar 5-8')[0]
            for row in first_block.splitlines():
                if '|' in row:
                    self.assertFalse(any(c.isdigit() for c in row.split('|', 1)[1]))
            pm = pretty_midi.PrettyMIDI(midis.split(';')[0])
            exported = [n for i in pm.instruments for n in i.notes]
            self.assertEqual(len(exported), 2)
            self.assertTrue(all(n.start >= 11 for n in exported))
            score = converter.parse(xmls.split(';')[0])
            sounding = list(score.flatten().notes)
            beat_seconds = 60 / opts.bpm
            attacks = [n for n in sounding if n.tie is None or n.tie.type == 'start']
            self.assertEqual(len(attacks), 2)
            self.assertTrue(all(float(n.offset)*beat_seconds >= 11 for n in sounding))

    def test_notation_rounding_cannot_move_entry_before_manual_time(self):
        from tabgen.sheet import export_musicxml
        from music21 import converter
        opts = TabOptions(lead_start_seconds=1.005, grid_offset=.01)
        arranged = arrange_melody([note(60, 1.01), note(62, 1.14)], opts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'lead.musicxml'
            export_musicxml(arranged, str(path), opts, start_floor=opts.lead_start_seconds)
            sounding = list(converter.parse(path).flatten().notes)
            self.assertTrue(all(float(n.offset)*.5 >= 1.005 for n in sounding))
            attacks = [n for n in sounding if n.tie is None or n.tie.type == 'start']
            self.assertEqual(len(attacks), 2)

    def test_empty_lead_exports_rests_without_a_fake_note(self):
        from tabgen.pipeline import generate
        import pretty_midi
        notes = intro()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'audio.wav'
            source.touch()
            with patch('tabgen.transcribe.transcribe', return_value=notes), \
                 patch('tabgen.harmony.audio_chord_changes', return_value=[]):
                text, midis, xmls, _ = generate(str(source), TabOptions(audio_lead_detection=False),
                                               split_lead_rhythm=True, auto_bpm=False)
            self.assertIn('リードなし／休符', text)
            pm = pretty_midi.PrettyMIDI(midis.split(';')[0])
            self.assertFalse(any(i.notes for i in pm.instruments))
            tree = ET.parse(xmls.split(';')[0])
            self.assertFalse(tree.findall('.//note/pitch'))
            self.assertTrue(tree.findall('.//note/rest'))


if __name__ == '__main__':
    unittest.main()
