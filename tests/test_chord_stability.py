import unittest

from test_analysis import features
from tabgen.harmony import chords_from_chroma
from tabgen.tab import TabOptions


class ChordStabilityTests(unittest.TestCase):
    def test_arpeggio_spread_over_beats_forms_one_chord(self):
        data = features([(i*.5, (i+1)*.5, [p]) for i,p in enumerate([48,52,55,60]*3)])
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C'])
        self.assertEqual(changes[0].end, 6.)

    def test_one_beat_extension_does_not_toggle_the_chord(self):
        data = features([(0, 1.5, [48,52,55]), (1.5, 2, [48,52,55,59]), (2, 4, [48,52,55])])
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C'])

    def test_sustained_seventh_is_kept(self):
        data = features([(0, 2, [48,52,55]), (2, 4, [48,52,55,59])])
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C','Cmaj7'])
        self.assertAlmostEqual(changes[1].start, 2.)

    def test_weak_partial_chord_does_not_create_nc_hole(self):
        data = features([(0, 1, [48,52,55]), (1, 1.5, [48]), (1.5, 3, [48,52,55])])
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C'])

    def test_real_rest_between_same_chords_is_not_filled(self):
        data = features([(0, 1, [48,52,55]), (1, 1.5, []), (1.5, 3, [48,52,55])])
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C','N.C.','C'])
        self.assertEqual((changes[1].start,changes[1].end), (1.,1.5))

    def test_common_tone_progression_does_not_collapse_to_one_chord(self):
        data = features([(0, 2, [48,52,55]), (2, 4, [45,48,52]),
                         (4, 6, [41,45,48]), (6, 8, [43,47,50])])
        for stability in (1,2,4):
            with self.subTest(stability=stability):
                changes = chords_from_chroma(data, TabOptions(chord_beats=stability))
                self.assertEqual([c.name for c in changes], ['C','Am','F','G'])
                self.assertEqual([c.start for c in changes], [0.,2.,4.,6.])

    def test_strong_one_beat_changes_survive_duration_preference(self):
        data = features([(0, .5, [48,52,55]), (.5, 1, [42,46,49]), (1, 1.5, [39,43,46])])
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C','F#','D#'])

    def test_long_hold_has_no_artificial_maximum_duration(self):
        changes = chords_from_chroma(features([(0, 20, [48,52,55])]), TabOptions())
        self.assertEqual(len(changes), 1)
        self.assertEqual((changes[0].name,changes[0].end), ('C',20.))


if __name__ == '__main__':
    unittest.main()
