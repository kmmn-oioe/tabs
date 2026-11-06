import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tabgen.transcribe import merge_transcriptions, transcribe


def note(pitch, start, end, confidence=.8):
    return dict(pitch=pitch, start=start, end=end, confidence=confidence)


class StereoTranscriptionTests(unittest.TestCase):
    def test_merge_deduplicates_agreement_but_preserves_repeated_attacks(self):
        primary = [note(60, 0, .5)]
        side = [note(60, .02, .5, .9), note(60, .3, .7), note(64, .1, .4)]
        merged = merge_transcriptions(primary, side)
        self.assertEqual([(n['pitch'], n['start']) for n in merged], [(60,.02),(64,.1),(60,.3)])
        self.assertEqual(primary[0]['start'], 0)

    def test_mid_and_side_both_supply_candidates_with_monotonic_progress(self):
        sr = 22050
        t = np.arange(sr)/sr
        mid = .2*np.sin(2*np.pi*220*t)
        side = .2*np.sin(2*np.pi*440*t)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stereo.wav'
            sf.write(path, np.column_stack([mid+side, mid-side]), sr)
            with patch('tabgen.transcribe._transcribe_mono', side_effect=[[note(57,0,1)], [note(69,0,1)]]) as predict:
                notes = transcribe(str(path), stereo_candidates=True)
                self.assertEqual(predict.call_count, 2)
                self.assertEqual(predict.call_args_list[0].args[2], (50,72.5))
                self.assertEqual(predict.call_args_list[1].args[2], (72.5,95))
            self.assertEqual({n['pitch'] for n in notes}, {57,69})

    def test_mono_and_centered_stereo_do_not_require_a_second_inference(self):
        sr = 22050
        wave = .2*np.sin(2*np.pi*220*np.arange(sr)/sr)
        for audio in (wave, np.column_stack([wave,wave])):
            with self.subTest(channels=audio.ndim), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'audio.wav'
                sf.write(path, audio, sr)
                with patch('tabgen.transcribe._transcribe_mono', return_value=[]) as predict:
                    transcribe(str(path), stereo_candidates=True)
                    predict.assert_called_once()


if __name__ == '__main__':
    unittest.main()
