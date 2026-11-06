import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from tabgen.tab import (TabOptions, playable, arrange_notes, assign_positions,
                        notes_to_tab, DIFFICULTIES)


def note(pitch, start=0., end=None):
    return dict(pitch=pitch, start=start, end=start + .4 if end is None else end, confidence=.9)


class PlayabilityTests(unittest.TestCase):
    def test_unique_strings_even_for_dense_chord(self):
        shape = assign_positions(list(range(48, 77)), 5, TabOptions(difficulty='advanced'))
        self.assertTrue(shape)
        self.assertEqual(len(shape), len({s for s, f in shape}))
        self.assertTrue(playable(tuple(shape), TabOptions(difficulty='advanced')))

    def test_stretch_open_strings_and_barre(self):
        opts = TabOptions(difficulty='advanced')
        self.assertFalse(playable(((0, 1), (1, 12)), opts))
        self.assertTrue(playable(((0, 0), (1, 12)), opts))
        # F barre: index plus three fretting fingers.
        f = ((0, 1), (1, 1), (2, 2), (3, 3), (4, 3), (5, 1))
        self.assertTrue(playable(f, opts))
        self.assertFalse(playable(((0, 1), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4)), opts))

    def test_phrase_lookahead_avoids_fast_jump(self):
        opts = TabOptions(difficulty='advanced')
        greedy = assign_positions([66], 2, opts)[0][1]
        plan = arrange_notes([note(66), note(81, .125)], opts)
        self.assertEqual(len(plan), 2)
        self.assertLess(abs(plan[0]['fret'] - plan[1]['fret']), abs(greedy - plan[1]['fret']))

    def test_difficulty_preserves_melody_and_limits_voices(self):
        chord = [note(p) for p in [40, 48, 55, 60, 64, 67]]
        counts = []
        for level, (limit, *_) in DIFFICULTIES.items():
            plan = arrange_notes(chord, TabOptions(difficulty=level))
            self.assertLessEqual(len(plan), limit)
            self.assertIn(67, [n['pitch'] for n in plan])
            counts.append(len(plan))
        self.assertLess(counts[0], counts[-1])

    def test_capo_tuning_and_input_preserved(self):
        notes = [note(42), note(66, .5)]
        original = [dict(n) for n in notes]
        opts = TabOptions(capo=2, tuning=[64, 59, 55, 50, 45, 38])
        plan = arrange_notes(notes, opts)
        self.assertEqual(len(plan), 2)
        for n in plan:
            self.assertEqual(n['pitch'], opts.tuning[n['string']] + opts.capo + n['fret'])
        self.assertEqual(notes, original)

    def test_release_previous_shape_and_filter(self):
        plan = arrange_notes([note(60, end=2), note(62, .5), note(127, 1)])
        self.assertEqual(len(plan), 2)
        self.assertLessEqual(plan[0]['end'], plan[1]['start'])
        self.assertEqual(arrange_notes([]), [])
        self.assertEqual(assign_positions([127], 0, TabOptions()), [])
        with self.assertRaises(ValueError):
            arrange_notes([], TabOptions(difficulty='invalid'))

    def test_pipeline_exports_the_arrangement(self):
        from tabgen.pipeline import generate
        import pretty_midi
        import xml.etree.ElementTree as ET
        notes = [note(p) for p in [40, 48, 55, 60, 64, 67]]
        for split in [False, True]:
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp)/'audio.wav'
                source.touch()
                with patch('tabgen.transcribe.transcribe', return_value=notes):
                    text, midis, xmls, _ = generate(str(source), TabOptions(difficulty='easy'),
                                                   auto_bpm=False, split_lead_rhythm=split)
                expected = []
                if split:
                    from tabgen.split import split_notes
                    from tabgen.rhythm import arrange_rhythm
                    lead, _ = split_notes(notes)
                    expected = [arrange_notes(lead, TabOptions(difficulty='easy')),
                                arrange_rhythm(notes, TabOptions(difficulty='easy'))[0]]
                else:
                    expected = [arrange_notes(notes, TabOptions(difficulty='easy'))]
                self.assertEqual(len(midis.split(';')), len(expected))
                self.assertEqual(len(xmls.split(';')), len(expected))
                for path, xml, part in zip(midis.split(';'), xmls.split(';'), expected):
                    actual = [n.pitch for i in pretty_midi.PrettyMIDI(path).instruments for n in i.notes]
                    self.assertEqual(sorted(actual), sorted(n['pitch'] for n in part))
                    self.assertEqual(len(ET.parse(xml).findall('.//note/pitch')), len(part))
                    self.assertIn(notes_to_tab(part, TabOptions(difficulty='easy'), arranged=True), text)


if __name__ == '__main__':
    unittest.main()
