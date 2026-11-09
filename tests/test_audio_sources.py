import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

import numpy as np
import soundfile as sf

from tabgen.pipeline import generate
from tabgen.audio import read_preview_audio, _ffmpeg_location
from tabgen.separate import guitar_backing_path
from tabgen.tab import TabOptions


class PipelineSourceTests(unittest.TestCase):
    def test_playback_uses_complement_stem_and_final_notes_with_either_transcription_source(self):
        # A conspicuous original signal must never become the backing.
        for use_separation in (False, True):
            with self.subTest(use_separation=use_separation), tempfile.TemporaryDirectory() as tmp:
                original = Path(tmp)/'original.wav'
                separated = Path(tmp)/'guitar.wav'
                backing = Path(tmp)/'no_guitar.wav'
                for path, value in ((original,.7),(separated,.4),(backing,.05)):
                    sf.write(path,np.full((44100,2),value),22050,subtype='FLOAT')
                notes = [dict(pitch=76,start=.38,end=.8,confidence=.9),
                         dict(pitch=52,start=.38,end=.8,confidence=.9)]
                paths,diagnostics = {},{}
                opts = TabOptions(bpm=120,rhythm_mode='notes',audio_lead_detection=False,
                                  suppress_chord_attacks=False)
                from tabgen.synth import render_tab_mix
                with patch('tabgen.transcribe.transcribe',return_value=notes) as transcribe, \
                     patch('tabgen.separate.demucs_available',return_value=True), \
                     patch('tabgen.separate.separate_guitar',return_value=str(separated)) as separate, \
                     patch('tabgen.harmony.audio_chord_changes',return_value=[]), \
                     patch('tabgen.rhythm_audio.analyze_rhythm_attacks',
                           side_effect=lambda path, notes: [dict(energy=.1,share=.5,novelty=.5) for n in notes]), \
                     patch('tabgen.synth.render_tab_mix',wraps=render_tab_mix) as render:
                    generate(str(original),opts,use_separation=use_separation,
                             split_lead_rhythm=True,auto_bpm=False,prepare_playback=True,
                             audio_sources=paths,diagnostics=diagnostics)
                separate.assert_called_once()
                self.assertEqual(transcribe.call_args.args[0],str(separated if use_separation else original))
                self.assertEqual(render.call_args.args[0],str(backing))
                self.assertEqual(render.call_args.args[1],[diagnostics['lead'],diagnostics['rhythm']])
                self.assertTrue(diagnostics['lead'])
                self.assertTrue(diagnostics['rhythm'])
                self.assertNotEqual(diagnostics['lead'][0]['start'],notes[0]['start'])
                data,sr = read_preview_audio(paths['playback'])
                self.assertEqual(len(data),44100)
                self.assertEqual(sr,22050)
                np.testing.assert_allclose(data[:2205],.04,atol=1e-7)
                self.assertGreater(np.std(data[12000:14000]),.05)
                self.assertNotIn('playback_error',paths)

    def test_missing_demucs_keeps_tab_but_does_not_offer_original_for_playback(self):
        notes = [dict(pitch=64,start=0.,end=.5,confidence=.9)]
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'original.wav'
            original.touch()
            paths={'playback':'stale.wav'}
            with patch('tabgen.transcribe.transcribe',return_value=notes), \
                 patch('tabgen.separate.demucs_available',return_value=False):
                result=generate(str(original),auto_bpm=False,prepare_playback=True,audio_sources=paths)
            self.assertTrue(result[0])
            self.assertNotIn('playback',paths)
            self.assertIn('Demucs',paths['playback_error'])

    def test_failed_playback_only_separation_keeps_transcription_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'original.wav'
            original.touch()
            paths={}
            with patch('tabgen.transcribe.transcribe',return_value=[]), \
                 patch('tabgen.separate.demucs_available',return_value=True), \
                 patch('tabgen.separate.separate_guitar',side_effect=RuntimeError('model unavailable')):
                generate(str(original),auto_bpm=False,prepare_playback=True,audio_sources=paths)
            self.assertNotIn('playback',paths)
            self.assertIn('model unavailable',paths['playback_error'])

    def test_other_stem_or_missing_complement_cannot_be_used_as_guitar_free_backing(self):
        with tempfile.TemporaryDirectory() as tmp:
            backing=Path(tmp)/'no_guitar.wav'
            backing.touch()
            with self.assertRaisesRegex(RuntimeError,'ギター専用'):
                guitar_backing_path(str(Path(tmp)/'other.wav'))
            self.assertEqual(guitar_backing_path(str(Path(tmp)/'guitar.wav')),str(backing))
            backing.unlink()
            with self.assertRaisesRegex(RuntimeError,'no_guitar.wav'):
                guitar_backing_path(str(Path(tmp)/'guitar.wav'))

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
