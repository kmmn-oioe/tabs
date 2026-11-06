import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from tabgen.rhythm import chord_changes, arrange_rhythm, progression_text
from tabgen.tab import TabOptions, playable, notes_to_tab, DIFFICULTIES


def chord(pitches, start, end):
    return [dict(pitch=p, start=start, end=end, confidence=.9) for p in pitches]


class RhythmTests(unittest.TestCase):
    def test_progression_and_sustained_voicings(self):
        notes = (chord([48, 52, 55], 0, 1) + chord([43, 47, 50], 1, 2)
                 + chord([45, 48, 52], 2, 3) + chord([41, 45, 48], 3, 4))
        for difficulty in DIFFICULTIES:
            with self.subTest(difficulty=difficulty):
                opts = TabOptions(difficulty=difficulty)
                arranged, changes = arrange_rhythm(notes, opts)
                self.assertEqual([c.name for c in changes], ['C', 'G', 'Am', 'F'])
                for i, change in enumerate(changes):
                    part = [n for n in arranged if n['start'] == i]
                    self.assertTrue(playable(tuple((n['string'], n['fret']) for n in part), opts))
                    self.assertGreaterEqual(len(part), 3)
                    self.assertEqual({n['end'] for n in part}, {i + 1})
                    self.assertEqual({n['pitch'] % 12 for n in part},
                                     {(change.root + x) % 12 for x in change.intervals})
                text = notes_to_tab(arranged, opts, arranged=True)
                self.assertIn('~~', text)
                self.assertIn('Bar 2-2', notes_to_tab(arranged, TabOptions(bars_per_line=1), arranged=True))

    def test_arpeggio_merges_identical_chords(self):
        notes = []
        for i, pitch in enumerate([48, 52, 55, 60] * 2):
            notes += chord([pitch], i * .25, i * .25 + .2)
        arranged, changes = arrange_rhythm(notes, TabOptions())
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].name, 'C')
        self.assertEqual({n['start'] for n in arranged}, {0})
        self.assertAlmostEqual(changes[0].end, 1.95)

    def test_silence_single_pitch_and_power_chord(self):
        notes = chord([48, 52, 55], 0, 1) + chord([50, 54, 57], 2, 3)
        arranged, changes = arrange_rhythm(notes, TabOptions())
        self.assertEqual([c.name for c in changes], ['C', 'N.C.', 'D'])
        self.assertFalse(any(n['start'] == 1 for n in arranged))
        self.assertEqual(chord_changes(chord([48], 0, 1), TabOptions())[0].name, 'N.C.')
        self.assertEqual(chord_changes(chord([48, 55], 0, 1), TabOptions())[0].name, 'C5')
        self.assertEqual(arrange_rhythm([], TabOptions()), ([], []))

    def test_interval_bpm_and_offset(self):
        notes = chord([48, 52, 55], .25, .75) + chord([43, 47, 50], .75, 1.25)
        changes = chord_changes(notes, TabOptions(chord_beats=1, bpm=120, grid_offset=.25))
        self.assertEqual([c.name for c in changes], ['C', 'G'])
        self.assertAlmostEqual(changes[0].start, .25)
        self.assertAlmostEqual(changes[1].start, .75)
        self.assertIn('1小節 2拍: G', progression_text(changes, TabOptions(grid_offset=.25)))
        with self.assertRaises(ValueError):
            chord_changes(notes, TabOptions(chord_beats=0))

    def test_seventh_chords_capo_and_custom_tuning(self):
        for suffix, pitches in [('7', [43, 47, 50, 53]), ('maj7', [48, 52, 55, 59]),
                                ('m7', [45, 48, 52, 55])]:
            for difficulty in DIFFICULTIES:
                opts = TabOptions(difficulty=difficulty, capo=2, tuning=[64, 59, 55, 50, 45, 38])
                arranged, changes = arrange_rhythm(chord(pitches, 0, 1), opts)
                self.assertEqual(changes[0].suffix, suffix)
                self.assertGreaterEqual(len(arranged), 3)
                for n in arranged:
                    self.assertEqual(n['pitch'], opts.tuning[n['string']] + opts.capo + n['fret'])

    def test_midi_and_musicxml_have_chord_duration_and_symbols(self):
        from tabgen.export import export_midi
        from tabgen.sheet import export_musicxml
        import pretty_midi
        opts = TabOptions()
        arranged, _ = arrange_rhythm(chord([48, 52, 55], 0, 2) + chord([43, 47, 50], 2, 4), opts)
        with TemporaryDirectory() as tmp:
            mid, xml = Path(tmp)/'rhythm.mid', Path(tmp)/'rhythm.musicxml'
            export_midi(arranged, str(mid))
            export_musicxml(arranged, str(xml), opts)
            actual = [n for i in pretty_midi.PrettyMIDI(str(mid)).instruments for n in i.notes]
            self.assertEqual({n.start for n in actual}, {0., 2.})
            self.assertEqual({n.end - n.start for n in actual}, {2.})
            self.assertEqual([e.text for e in ET.parse(xml).findall('.//harmony/root/root-step')], ['C', 'G'])


if __name__ == '__main__':
    unittest.main()
