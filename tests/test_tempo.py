import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tabgen.tempo import (_onset_envelope, estimate_bpm, estimate_offset,
                          estimate_bpm_and_offset)


def pulse_envelope(bpm, *, duration=24., offset=.173, accent=0., dt=.01):
    times=np.arange(0,duration,dt)
    envelope=np.zeros(len(times))
    for i,at in enumerate(np.arange(offset,duration,30/bpm)):
        weight=.5+(1 if i%2==0 else 0)+(accent if i%3==0 else 0)
        envelope+=weight*np.exp(-.5*((times-at)/.015)**2)
    return envelope,dt


def click_audio(bpm, *, sr=22050, duration=18., offset=.173, eighths=True):
    audio=np.zeros(round(duration*sr))
    rng=np.random.default_rng(42)
    t=np.arange(round(sr*.035))/sr
    for step,at in enumerate(np.arange(offset,duration,60/bpm/(2 if eighths else 1))):
        if eighths:
            level=[1,.65,.85,.65][(step//2)%4] if step%2==0 else .3
        else:
            level=[1,.65,.85,.65][step%4]
        wave=(np.sin(2*np.pi*(240 if step%2==0 else 1800)*t)+.2*rng.standard_normal(len(t)))
        wave*=np.exp(-t*120)*level
        first=round(at*sr);last=min(len(audio),first+len(t))
        audio[first:last]+=wave[:last-first]
    return audio


class TempoTests(unittest.TestCase):
    def test_three_eighth_accent_does_not_replace_the_quarter_beat(self):
        # Previously the autocorrelation maximum selected 100, then stopped.
        # The underlying binary grid is 150; the dotted accent repeats at 100.
        env,dt=pulse_envelope(150,accent=1.5)
        self.assertAlmostEqual(estimate_bpm(env,dt),150,delta=.15)

    def test_different_slow_and_medium_tempos_with_quarters_and_eighths(self):
        for bpm in [64,72,90,107.7,123.4,147]:
            for eighths in [False,True]:
                with self.subTest(bpm=bpm,eighths=eighths):
                    env,dt=_onset_envelope(click_audio(bpm,eighths=eighths),22050)
                    predicted=estimate_bpm(env,dt)
                    self.assertAlmostEqual(predicted,bpm,delta=.2)
                    phase=estimate_offset(env,dt,predicted)
                    beat=60/bpm
                    error=abs((phase-.173+beat/2)%beat-beat/2)
                    self.assertLess(error,.03)

    def test_fractional_tempo_remains_aligned_over_a_long_clip(self):
        env,dt=pulse_envelope(123.4,duration=80)
        predicted=estimate_bpm(env,dt)
        phase=estimate_offset(env,dt,predicted)
        self.assertAlmostEqual(predicted,123.4,delta=.08)
        self.assertLess(abs(phase-.173),.02)
        self.assertLess(80*abs(predicted-123.4)/123.4,.05)

    def test_requested_fast_range_is_not_folded_into_80_to_180(self):
        env,dt=_onset_envelope(click_audio(205),22050)
        self.assertAlmostEqual(estimate_bpm(env,dt,min_bpm=185,max_bpm=220),205,delta=.2)

    def test_no_input_or_constant_envelope_has_no_supported_phase(self):
        for env in [np.array([]),np.zeros(300),np.ones(300),np.array([0.,1.])]:
            with self.subTest(length=len(env)):
                self.assertEqual(estimate_bpm(env,.01),120)
                self.assertEqual(estimate_offset(env,.01,120),0)

    def test_short_audio_and_silence_do_not_raise_stft_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'silence.wav'
            for samples in [0,1,100,1024,22050*3]:
                sf.write(path,np.zeros(samples),22050)
                diagnostics={}
                self.assertEqual(estimate_bpm_and_offset(str(path),diagnostics=diagnostics),(120.,0.))
                self.assertTrue(diagnostics['fallback'])
                self.assertEqual(diagnostics['candidates'],[])

    def test_file_estimate_and_diagnostics_use_the_same_rounded_tempo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'click.wav'
            sf.write(path,click_audio(123.4),22050)
            diagnostics={}
            bpm,phase=estimate_bpm_and_offset(str(path),diagnostics=diagnostics)
        self.assertAlmostEqual(bpm,123.4,delta=.2)
        self.assertEqual(diagnostics['bpm'],bpm)
        self.assertEqual(diagnostics['grid_offset'],phase)
        self.assertEqual(diagnostics['method'],'pulse_grid')
        self.assertFalse(diagnostics['fallback'])
        self.assertGreater(len(diagnostics['candidates']),1)
        self.assertTrue(all(60 <= c['bpm'] <= 220 for c in diagnostics['candidates']))

    def test_envelope_is_not_modified_by_candidate_search(self):
        env,dt=pulse_envelope(120)
        original=env.copy()
        estimate_bpm(env,dt)
        estimate_offset(env,dt,120)
        np.testing.assert_array_equal(env,original)

    def test_pipeline_uses_estimated_tempo_for_quantization_and_saved_tab(self):
        import json
        from tabgen.pipeline import generate
        from tabgen.tab import TabOptions
        from tabgen.tabpdf import data_path
        notes=[dict(pitch=60+i,start=.173+i*.5,end=.4+i*.5,confidence=.9) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'click.wav'
            sf.write(path,click_audio(120),22050)
            options=TabOptions(bpm=60)
            diagnostics={}
            with patch('tabgen.transcribe.transcribe',return_value=notes):
                text,_,sheet,_=generate(str(path),options,diagnostics=diagnostics)
            saved=json.loads(data_path(sheet).read_text())
        self.assertAlmostEqual(options.bpm,120,delta=.2)
        self.assertIn(f'BPM {options.bpm:g}',text)
        self.assertEqual(saved['bpm'],options.bpm)
        self.assertEqual(diagnostics['tempo']['bpm'],options.bpm)
        grid=options.quantize*60/options.bpm
        for n in diagnostics['lead']:
            cell=(n['start']-options.grid_offset)/grid
            self.assertAlmostEqual(cell,round(cell))

    def test_invalid_parameters_are_rejected(self):
        env,_=pulse_envelope(120)
        for lo,hi in [(0,100),(150,100),(60,np.inf)]:
            with self.assertRaises(ValueError):
                estimate_bpm(env,.01,lo,hi)
        for dt in [0,-1,np.nan]:
            with self.assertRaises(ValueError):
                estimate_bpm(env,dt)
        with self.assertRaises(ValueError):
            estimate_bpm(np.array([1.,np.nan]*100),.01)
        with self.assertRaises(ValueError):
            estimate_offset(env,.01,0)


if __name__ == '__main__':
    unittest.main()
