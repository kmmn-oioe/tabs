import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tabgen.synth import render_tab_mix


class TabSynthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backing = str(Path(self.temp.name)/'no_guitar.wav')
        self.output = str(Path(self.temp.name)/'playback.wav')
        self.sr = 22050

    def render(self, backing, parts):
        sf.write(self.backing, backing, self.sr, subtype='FLOAT')
        render_tab_mix(self.backing, parts, self.output)
        return sf.read(self.output, dtype='float32', always_2d=True)

    def amplitude(self, data, hz):
        times = np.arange(len(data))/self.sr
        return abs(np.dot(data, np.exp(-2j*np.pi*hz*times))) * 2/len(data)

    def test_both_parts_use_tab_pitch_timing_and_silent_rests(self):
        parts = [[dict(pitch=69, start=.25, end=.75)],
                 [dict(pitch=57, start=.5, end=1.)]]
        data, sr = self.render(np.zeros((self.sr*2, 2)), parts)
        self.assertEqual(sr, self.sr)
        self.assertEqual(data.shape, (self.sr*2, 2))
        np.testing.assert_array_equal(data[:round(.25*sr)], 0)
        np.testing.assert_array_equal(data[sr:], 0)
        early = data[round(.3*sr):round(.45*sr), 0]
        overlap = data[round(.55*sr):round(.7*sr), 0]
        late = data[round(.8*sr):round(.95*sr), 0]
        self.assertGreater(self.amplitude(early, 440), .12)
        self.assertLess(self.amplitude(early, 220), .005)
        self.assertGreater(self.amplitude(overlap, 440), .12)
        self.assertGreater(self.amplitude(overlap, 220), .09)
        self.assertLess(self.amplitude(late, 440)/self.amplitude(late, 220), .3)
        np.testing.assert_array_equal(data[:, 0], data[:, 1])

    def test_only_supplied_backing_is_audible_during_tab_rests(self):
        backing = np.tile([.125, -.25], (self.sr, 1))
        data, _ = self.render(backing, [[dict(pitch=69, start=.25, end=.5)]])
        np.testing.assert_allclose(data[:self.sr//4], backing[:self.sr//4]*.8, atol=1e-7)
        np.testing.assert_allclose(data[self.sr//2:], backing[self.sr//2:]*.8, atol=1e-7)
        self.assertGreater(np.std(data[self.sr//4:self.sr//2, 0]), .08)

    def test_polyphonic_mix_does_not_clip_and_inputs_are_unchanged(self):
        notes = [dict(pitch=p, start=0., end=.5) for p in range(40, 88)]
        data, _ = self.render(np.full((self.sr, 2), .9), [notes])
        self.assertLessEqual(float(np.max(abs(data))), .980001)
        np.testing.assert_allclose(sf.read(self.backing)[0], .9, atol=1e-7)
        self.assertEqual(len(notes), 48)

    def test_empty_tab_keeps_backing_duration_and_stereo(self):
        backing = np.tile([.125, .25], (self.sr, 1))
        data, _ = self.render(backing, [[], []])
        np.testing.assert_allclose(data, backing*.8, atol=1e-7)

    def test_last_tab_note_is_not_cut_off_at_backing_end(self):
        data, _ = self.render(np.zeros(self.sr//2), [[dict(pitch=69, start=.4, end=1.)]])
        self.assertEqual(data.shape, (self.sr, 1))
        self.assertGreater(np.std(data[round(.6*self.sr):round(.8*self.sr)]), .08)

    def test_invalid_note_times_are_rejected(self):
        for start, end in [(float('nan'), 1.), (0., float('inf')), (.5, .4)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    self.render(np.zeros(self.sr), [[dict(pitch=69, start=start, end=end)]])
