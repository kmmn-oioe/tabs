import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

import numpy as np
import soundfile as sf

from tabgen.pipeline import generate
from tabgen.audio import read_preview_audio, _ffmpeg_location


class PipelineSourceTests(unittest.TestCase):
    def test_original_and_separated_paths_are_returned_separately(self):
        notes = [dict(pitch=64,start=0.,end=.5,confidence=.9)]
        with tempfile.TemporaryDirectory() as tmp:
            original,separated = Path(tmp)/'original.wav',Path(tmp)/'guitar.wav'
            original.touch()
            separated.touch()
            paths = {}
            with patch('tabgen.transcribe.transcribe',return_value=notes), \
                 patch('tabgen.separate.demucs_available',return_value=True), \
                 patch('tabgen.separate.separate_guitar',return_value=str(separated)):
                result = generate(str(original),use_separation=True,auto_bpm=False,audio_sources=paths)
            self.assertEqual(len(result),4)
            self.assertEqual(result[3],str(separated))
            self.assertEqual(paths,{'original':str(original),'separated':str(separated)})

    def test_skipped_separation_does_not_keep_a_stale_separated_source(self):
        notes = [dict(pitch=64,start=0.,end=.5,confidence=.9)]
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp)/'original.wav'
            original.touch()
            paths = {'separated':'old_track.wav'}
            with patch('tabgen.transcribe.transcribe',return_value=notes), \
                 patch('tabgen.separate.demucs_available',return_value=False):
                generate(str(original),use_separation=True,auto_bpm=False,audio_sources=paths)
            self.assertEqual(paths,{'original':str(original)})

    def test_original_m4a_can_be_decoded_for_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav,m4a = Path(tmp)/'original.wav',Path(tmp)/'original.m4a'
            t = np.arange(4410)/22050
            sf.write(wav,np.column_stack([.1*np.sin(2*np.pi*220*t),.2*np.sin(2*np.pi*330*t)]),22050)
            subprocess.run([_ffmpeg_location() or 'ffmpeg','-v','error','-i',str(wav),'-c:a','aac',str(m4a)],check=True)
            data,sr = read_preview_audio(str(m4a))
            self.assertEqual(sr,22050)
            self.assertEqual(data.shape[1],2)
            self.assertGreater(len(data),4000)
            self.assertGreater(float(np.sqrt(np.mean(data[:,1]**2))),float(np.sqrt(np.mean(data[:,0]**2))))

    def test_unreadable_source_does_not_play_other_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)/'bad.wav'
            bad.touch()
            with patch('tabgen.audio.subprocess.run',return_value=subprocess.CompletedProcess([],1,b'',b'bad input')):
                with self.assertRaisesRegex(RuntimeError,'デコードに失敗'):
                    read_preview_audio(str(bad))


if __name__ == '__main__':
    unittest.main()
