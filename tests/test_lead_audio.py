import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_fixtures import SAMPLE_RATE, guitar_scene, recurring_melody_scene
from tabgen.audio import load_audio, load_audio_channels
from tabgen.lead_audio import analyze_lead_audio, _opposed_voice_evidence, NoteAudioFeatures
from tabgen.split import split_notes
from tabgen.tab import TabOptions


class LeadAudioTests(unittest.TestCase):
    def analyze(self, **kwargs):
        audio, notes, labels = guitar_scene(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scene.wav'
            sf.write(path, audio, SAMPLE_RATE)
            evidence = analyze_lead_audio(str(path), notes)
        return evidence, notes, labels

    def test_arpeggio_no_false_lead_even_though_note_tracker_finds_a_line(self):
        evidence, notes, _ = self.analyze()
        self.assertTrue(split_notes(notes)[0])  # Reproduces the old false positive.
        self.assertTrue(evidence.stereo)
        self.assertTrue(evidence.reference_found)
        lead, rhythm = split_notes(notes, lead_scores=evidence.scores)
        self.assertEqual(lead, [])
        self.assertEqual(len(rhythm), len(notes))

    def test_second_panned_voice_retained_and_backing_rejected(self):
        for spread in (.7, .45):
            with self.subTest(spread=spread):
                evidence, notes, labels = self.analyze(lead=True, spread=spread)
                detected = [score >= .6 for score in evidence.scores]
                self.assertEqual(detected, labels)
                lead, _ = split_notes(notes, lead_scores=evidence.scores)
                self.assertEqual([n['pitch'] for n in lead], [66,64,62,64,66])
                self.assertTrue(all(n['start'] >= 1.6 for n in lead))

    def test_downmix_loses_evidence_instead_of_faking_stereo_information(self):
        stereo, _, labels = self.analyze(lead=True)
        mono, _, _ = self.analyze(lead=True, stereo=False)
        self.assertFalse(mono.stereo)
        self.assertIn('定位情報なし', mono.summary())
        self.assertGreater(sum(s for s,l in zip(stereo.scores, labels) if l),
                           sum(s for s,l in zip(mono.scores, labels) if l) + 1)

    def test_moving_one_source_is_not_an_independent_melody(self):
        evidence, _, _ = self.analyze(moving=True)
        self.assertFalse(any(s >= .6 for s in evidence.scores))

    def test_pitch_dependent_timbre_does_not_invent_a_second_stereo_source(self):
        evidence, _, _ = self.analyze(varying_timbre=True)
        self.assertFalse(any(s >= .6 for s in evidence.scores))

    def test_phase_reveals_centered_voice_that_mono_average_cancels(self):
        evidence, notes, labels = self.analyze(lead=True, spread=0, lead_antiphase=True)
        self.assertEqual([s >= .6 for s in evidence.scores], labels)
        self.assertTrue(all(f.side_fraction > .7 for f,l in zip(evidence.features, labels) if l))
        lead, _ = split_notes(notes, lead_scores=evidence.scores)
        self.assertEqual([n['pitch'] for n in lead], [66,64,62,64,66])

    def test_distinct_harmonic_signature_supports_mono_evidence(self):
        evidence, _, labels = self.analyze(lead=True, stereo=False,
                                           lead_harmonics=(.55,1,.7,.5,.3))
        detected = [score >= .6 for score in evidence.scores]
        self.assertFalse(any(d for d,l in zip(detected, labels) if not l))
        self.assertGreaterEqual(sum(d for d,l in zip(detected, labels) if l), 3)

    def test_channel_loader_preserves_antiphase_and_resamples_each_channel(self):
        t = np.arange(44100) / 44100
        wave = .2*np.sin(2*np.pi*220*t)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'phase.wav'
            sf.write(path, np.column_stack([wave,-wave]), 44100, subtype='FLOAT')
            data, sr = load_audio_channels(str(path))
            mono, _ = load_audio(str(path))
        self.assertEqual(sr, 22050)
        self.assertEqual(data.shape, (22050, 2))
        self.assertGreater(np.max(abs(data)), .1)
        self.assertLess(np.max(abs(mono)), 1e-6)

    def test_silence_and_low_confidence_provide_no_source_evidence(self):
        _, notes, _ = guitar_scene()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'silence.wav'
            sf.write(path, np.zeros((SAMPLE_RATE*4, 2)), SAMPLE_RATE)
            evidence = analyze_lead_audio(str(path), notes)
        self.assertFalse(evidence.reference_found)
        self.assertFalse(any(evidence.scores))
        audio, notes, _ = guitar_scene(lead=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'quiet_detections.wav'
            sf.write(path, audio, SAMPLE_RATE)
            evidence = analyze_lead_audio(str(path), notes, min_confidence=1.)
        self.assertFalse(any(evidence.scores))
        self.assertFalse(evidence.reference_found)

    def test_pipeline_passes_audio_evidence_to_tabs_and_midi(self):
        from tabgen.pipeline import generate
        import pretty_midi
        for has_lead in (False, True):
            with self.subTest(lead=has_lead), tempfile.TemporaryDirectory() as tmp:
                audio, notes, _ = guitar_scene(lead=has_lead)
                path = Path(tmp)/'scene.wav'
                sf.write(path, audio, SAMPLE_RATE)
                with patch('tabgen.transcribe.transcribe', return_value=notes), \
                     patch('tabgen.harmony.audio_chord_changes', return_value=[]):
                    diagnostics = {}
                    text, midis, _, _ = generate(str(path), split_lead_rhythm=True, auto_bpm=False,
                                                diagnostics=diagnostics)
                self.assertIn('リード音声判定: ステレオ', text)
                self.assertEqual(len(diagnostics['lead_audio_scores']),len(notes))
                self.assertEqual(diagnostics['audio_supported'],
                                 [n for n,s in zip(notes,diagnostics['lead_audio_scores']) if s >= .6])
                midi = pretty_midi.PrettyMIDI(midis.split(';')[0])
                result = [n for i in midi.instruments for n in i.notes]
                if has_lead:
                    self.assertEqual(len(result), 5)
                    self.assertTrue(all(n.start >= 1.6 for n in result))
                else:
                    self.assertEqual(result, [])
                    self.assertIn('リードを確認できず／休符', text)

    def test_failed_audio_analysis_withholds_instead_of_reverting_to_false_lead(self):
        from tabgen.pipeline import generate
        _, notes, _ = guitar_scene()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.wav'
            path.touch()
            with patch('tabgen.transcribe.transcribe', return_value=notes), \
                 patch('tabgen.harmony.audio_chord_changes', return_value=[]):
                text, _, _, _ = generate(str(path), split_lead_rhythm=True, auto_bpm=False)
            self.assertIn('リード音声判定に失敗したため保留', text)
            self.assertIn('リードを確認できず／休符', text)

    def test_evidence_must_align_with_notes(self):
        with self.assertRaises(ValueError):
            split_notes([], lead_scores=[1.])

    def test_frequent_melody_is_not_misidentified_as_backing(self):
        for lower in (False, True):
            for mirror in (False, True):
                with self.subTest(lower=lower,mirror=mirror), tempfile.TemporaryDirectory() as tmp:
                    audio,notes,labels = recurring_melody_scene(lower_lead=lower,mirror=mirror)
                    path = Path(tmp)/'recurring.wav'
                    sf.write(path,audio,SAMPLE_RATE)
                    evidence = analyze_lead_audio(str(path),notes)
                    supported = [s >= .6 for s in evidence.scores]
                    self.assertEqual(evidence.method,'spatial_phrase')
                    self.assertFalse(any(s for s,l in zip(supported,labels) if not l))
                    self.assertGreaterEqual(sum(s for s,l in zip(supported,labels) if l),12)
                    lead,_ = split_notes(notes,lead_scores=evidence.scores)
                    self.assertTrue(all(n['start'] >= .8 for n in lead))
                    self.assertGreaterEqual(len(lead),12)

    def test_later_voice_change_does_not_relabel_the_intro(self):
        first,first_notes,first_labels = recurring_melody_scene()
        later,later_notes,_ = recurring_melody_scene(lower_lead=True,mirror=True)
        audio = np.concatenate([first,np.zeros((SAMPLE_RATE*4,2)),later,later,later])
        notes = list(first_notes)
        for shift in [8,12,16]:
            notes += [dict(n,start=n['start']+shift,end=n['end']+shift) for n in later_notes]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'changing_sources.wav'
            sf.write(path,audio,SAMPLE_RATE)
            e = analyze_lead_audio(str(path),notes)
        intro_scores = e.scores[:len(first_notes)]
        self.assertGreaterEqual(sum(s >= .6 for s,l in zip(intro_scores,first_labels) if l),12)
        self.assertFalse(any(s >= .6 for s,l in zip(intro_scores,first_labels) if not l))

    def test_rank_one_stereo_does_not_establish_two_phrase_sources(self):
        _,notes,labels = recurring_melody_scene()
        features = [NoteAudioFeatures(.7 if l else -.7,np.array([1.,0,0,0,0]),1.,1.,True,
                                     spatial_rank=.02) for l in labels]
        self.assertIsNone(_opposed_voice_evidence(notes,features))

    def test_competing_melodic_cluster_does_not_veto_the_supported_lead(self):
        # Two near-tied melodic signatures plus persistent backing. The old
        # all-or-nothing role margin withheld both melodic streams.
        for mirror in (1,-1):
            for low_lead in (False,True):
                with self.subTest(mirror=mirror,low_lead=low_lead):
                    notes,features=[],[]
                    def add(pitch,start,pan,confidence):
                        notes.append(dict(pitch=pitch,start=start,end=start+.3,confidence=confidence))
                        features.append(NoteAudioFeatures(pan*mirror,np.array([1.,0,0,0,0]),
                                                          1.,1.,True,spatial_rank=.4))
                    for i in range(16):
                        add(80 if low_lead else 45,i*.25,-.75,.7)
                    expected=[52,54,61,59] if low_lead else [64,66,73,71]
                    first=len(notes)
                    for i in range(12):
                        add(expected[i%4],.5+i*.25,.75,.9)
                    for i in range(8):
                        add(expected[i%4]-12,.5+i*.25,0.,.35)
                    scores=_opposed_voice_evidence(notes,features)
                    self.assertIsNotNone(scores)
                    self.assertTrue(all(s >= .6 for s in scores[first:first+12]))
                    self.assertFalse(any(s >= .6 for s in scores[:first]))
                    lead,_=split_notes(notes,lead_scores=scores)
                    self.assertEqual([n['pitch'] for n in lead],expected*3)

    def test_tied_voices_without_independent_backing_are_still_ambiguous(self):
        notes,features=[],[]
        for pan in (-.75,.75):
            for i,pitch in enumerate([60,62,69,67]*2):
                notes.append(dict(pitch=pitch,start=i*.25,end=i*.25+.2,confidence=.8))
                features.append(NoteAudioFeatures(pan,np.array([1.,0,0,0,0]),1.,1.,True,
                                                  spatial_rank=.4))
        self.assertIsNone(_opposed_voice_evidence(notes,features))


if __name__ == '__main__':
    unittest.main()
