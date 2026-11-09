from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_fixtures import guitar_scene, SAMPLE_RATE
from tabgen.rhythm_audio import (analyze_rhythm_attacks, filter_rhythm_attacks,
                                  align_rhythm_grid, rhythm_voice_checks, recover_lead_holes)
from tabgen.lead_audio import NoteAudioFeatures
from tabgen.rhythm import arrange_rhythm
from tabgen.tab import TabOptions, arrange_notes, arrange_melody
from benchmarks.audit_rhythm import audit_rhythm


class RhythmAudioTests(unittest.TestCase):
    def test_confirmed_lead_with_fresh_attack_fills_only_an_empty_cell(self):
        opts=TabOptions(bpm=120)
        raw=[dict(pitch=p,start=t,end=t+.2,confidence=.9) for p,t in ((67,0.),(71,.5))]
        lead=arrange_melody(raw,opts)
        candidate=dict(pitch=72,start=.25,end=.45,confidence=.8)
        result,recovered=recover_lead_holes(lead,[candidate],[dict(reject=True,attack=.8)],opts)
        self.assertEqual([(n['pitch'],n['start']) for n in result],[(67,0.),(72,.25),(71,.5)])
        self.assertEqual([n for n in result if n not in recovered],lead)
        self.assertEqual(candidate['start'],.25)

    def test_handoff_does_not_overwrite_melody_or_create_notes_from_ambiguous_attack(self):
        opts=TabOptions(bpm=120)
        candidate=dict(pitch=72,start=.25,end=.45,confidence=.8)
        lead=arrange_melody([dict(candidate,pitch=67)],opts)
        self.assertEqual(recover_lead_holes(lead,[candidate],[dict(reject=True,attack=.8)],opts)[0],lead)
        for reject,attack in ((False,.9),(True,.1)):
            self.assertEqual(recover_lead_holes([],[candidate],[dict(reject=reject,attack=attack)],opts),([],[]))
        opts.lead_start_seconds=.5
        self.assertEqual(recover_lead_holes([],[candidate],[dict(reject=True,attack=.8)],opts),([],[]))
        opts.lead_start_seconds=.3
        lead=arrange_melody([dict(candidate,start=.37,pitch=67,end=.6)],opts)
        self.assertEqual(recover_lead_holes(lead,[dict(candidate,start=.31)],
                                         [dict(reject=True,attack=.8)],opts)[0],lead)

    def voice_scene(self, mirror=1, lower_lead=False):
        def note(pitch,start):
            return dict(pitch=pitch,start=start,end=start+.3,confidence=.8)
        def feature(pan):
            h=np.array([1.,.3,.1,0.,0.]);h/=np.linalg.norm(h)
            return NoteAudioFeatures(pan*mirror,h,1.,.9,True,spatial_rank=.5)
        backing=[note(72 if lower_lead else 48,i*.4) for i in range(6)]
        lead=[note(52 if lower_lead else 76,i*.4) for i in range(6)]
        # Include real octave accompaniment; pitch height must not decide role.
        notes=backing+[note(84 if lower_lead else 60,1.),note(55 if lower_lead else 79,1.)]
        return notes,[feature(-.8) for n in notes[:-1]]+[feature(.8)],lead,[feature(.8) for n in lead]

    def test_lead_residue_is_withheld_but_real_octave_backing_is_kept(self):
        for mirror in (-1,1):
            for lower_lead in (False,True):
                with self.subTest(mirror=mirror,lower_lead=lower_lead):
                    notes,features,lead,lf=self.voice_scene(mirror,lower_lead)
                    checks=rhythm_voice_checks(notes,features,lead,lf)
                    self.assertEqual([c['reject'] for c in checks],[False]*7+[True])

    def test_pan_alone_or_missing_backing_does_not_establish_lead_ownership(self):
        notes,features,lead,lf=self.voice_scene()
        self.assertFalse(rhythm_voice_checks(notes,features,[],[])[-1]['reject'])
        self.assertFalse(rhythm_voice_checks(notes[-1:],features[-1:],lead,lf)[-1]['reject'])
        features[-1].harmonics=np.array([0.,0.,0.,1.,0.])
        self.assertFalse(rhythm_voice_checks(notes,features,lead,lf)[-1]['reject'])

    def test_mono_ambiguous_channels_and_unbracketed_candidate_are_preserved(self):
        notes,features,lead,lf=self.voice_scene()
        for f in features+lf:
            f.spatial_rank=0.
        self.assertFalse(any(c['reject'] for c in rhythm_voice_checks(notes,features,lead,lf)))
        notes,features,lead,lf=self.voice_scene()
        notes[-1]['start']=0.
        self.assertFalse(rhythm_voice_checks(notes,features,lead,lf)[-1]['reject'])

    def test_post_selection_check_does_not_change_other_notes_or_voicings(self):
        opts=TabOptions(rhythm_mode='notes')
        notes=[dict(pitch=p,start=t,end=t+.3,confidence=.9)
               for t in (0.,.5) for p in (52,55,59,64)]
        before,_=arrange_rhythm(notes,opts,changes=[])
        target=before[0]
        flagged=[dict(n,_audio_rejection='confirmed_lead_source')
                 if n['pitch']==target['pitch'] and n['start']==target['start'] else n for n in notes]
        after,_=arrange_rhythm(flagged,opts,changes=[])
        self.assertEqual(after,[n for n in before if n!=target])
        self.assertFalse(any('_audio_rejection' in n for n in after))
        diagnostics=dict(options=asdict(opts),transcribed=notes,rhythm_candidates=notes,
                         rhythm_arrangement_input=flagged,rhythm=after)
        audit=audit_rhythm(dict(id='fixture',evaluation_window=[0,1],notes=[]),diagnostics)
        self.assertTrue(audit['provenance_replay_identical'])

    def test_quiet_arpeggio_retains_the_same_attacks_as_loud_arpeggio(self):
        audio, notes, _ = guitar_scene()
        for gain in (1., .001):
            with self.subTest(gain=gain), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'arpeggio.wav'
                sf.write(path, audio*gain, SAMPLE_RATE, subtype='FLOAT')
                attacks = analyze_rhythm_attacks(str(path), notes)
                kept = filter_rhythm_attacks(notes, attacks)
                self.assertEqual(kept, notes)

    def test_silence_does_not_confirm_detected_pitches(self):
        notes = [dict(pitch=64, start=.25, end=.5, confidence=.9)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'silence.wav'
            sf.write(path, np.zeros(SAMPLE_RATE), SAMPLE_RATE)
            attacks = analyze_rhythm_attacks(str(path), notes)
        self.assertEqual(filter_rhythm_attacks(notes, attacks), [])

    def test_rearticulated_same_pitch_is_retained_and_ringing_fundamental_is_deferred(self):
        sr = SAMPLE_RATE
        audio = np.zeros(sr)
        for start in (.1, .5):
            t = np.arange(round(.35*sr))/sr
            audio[round(start*sr):round(start*sr)+len(t)] += np.sin(2*np.pi*440*t)*np.minimum(1,t/.005)*np.exp(-5*t)
        notes = [dict(pitch=69, start=s, end=s+.3, confidence=.9) for s in (.1,.5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'picked.wav'
            sf.write(path, audio, sr, subtype='FLOAT')
            ringing = dict(pitch=69,start=.3,end=.4,confidence=.9)
            attacks = analyze_rhythm_attacks(str(path), notes+[ringing])
        self.assertEqual(filter_rhythm_attacks(notes+[ringing], attacks), notes+[ringing])
        self.assertTrue(all(a['novelty']>.05 for a in attacks[:2]))
        self.assertLess(attacks[-1]['novelty'],.05)

    def test_mismatched_evidence_fails_instead_of_filtering_another_note(self):
        with self.assertRaises(ValueError):
            filter_rhythm_attacks([dict(pitch=60)], [])

    def test_grid_delay_is_measured_without_changing_tempo_and_rest_pattern(self):
        opts = TabOptions(bpm=120, rhythm_mode='notes')
        notes = [dict(pitch=52, start=i*.25+.03, end=i*.25+.18, confidence=.9)
                 for i in range(20) if i not in (8,9)]
        # One late detection would otherwise snap to the following 16th.
        notes[6]['start'] = 6*.25+.075
        adjusted, info = align_rhythm_grid(notes, opts)
        arranged = arrange_notes(adjusted, opts)
        self.assertTrue(info['applied'])
        self.assertEqual([n['start'] for n in arranged], [i*.25 for i in range(20) if i not in (8,9)])
        self.assertEqual(opts.bpm,120)
        self.assertEqual(opts.grid_offset,0)
        self.assertEqual(notes[6]['start'],1.575)
        self.assertFalse(any('_snap_start' in n for n in arranged))

    def test_sparse_or_unconcentrated_onsets_do_not_calibrate_a_delay(self):
        opts=TabOptions()
        for times in ([.28,.53], np.arange(30)*.157):
            notes=[dict(pitch=52,start=float(s),end=float(s+.1),confidence=.9) for s in times]
            result,info=align_rhythm_grid(notes,opts)
            self.assertFalse(info['applied'])
            self.assertEqual(result,notes)

    def test_on_grid_sixteenth_notes_and_syncopation_stay_on_their_original_grid(self):
        opts=TabOptions(bpm=120)
        notes=[dict(pitch=52,start=i*.125,end=i*.125+.1,confidence=.9)
               for i in range(24) if i%4!=0]
        adjusted,info=align_rhythm_grid(notes,opts)
        self.assertFalse(info['applied'])
        self.assertEqual([n['start'] for n in arrange_notes(adjusted,opts)], [n['start'] for n in notes])

    def test_audit_traces_filtered_and_retimed_notes_to_original_candidates(self):
        opts=TabOptions(bpm=120,rhythm_mode='notes')
        raw=[dict(pitch=52,start=i*.25+.03,end=i*.25+.2,confidence=.9) for i in range(20)]
        adjusted,_=align_rhythm_grid(raw,opts)
        d=dict(options=asdict(opts),transcribed=raw,rhythm_candidates=raw,
               rhythm_arrangement_input=adjusted,rhythm=arrange_notes(adjusted,opts))
        r=audit_rhythm(dict(id='fixture',evaluation_window=[0,5],notes=[]),d)
        self.assertEqual(len(r['extras']),20)
        self.assertTrue(r['provenance_replay_identical'])

    def test_audit_distinguishes_audio_filter_loss_from_arrangement_loss(self):
        note=dict(pitch=52,start=0.,end=.2,confidence=.9)
        opts=TabOptions(rhythm_mode='notes')
        d=dict(options=asdict(opts),transcribed=[note],rhythm_candidates=[note],
               rhythm_audio_supported=[],rhythm_arrangement_input=[],rhythm=[])
        case=dict(id='fixture',evaluation_window=[0,1],notes=[dict(note,part='rhythm',id='rhythm.0')])
        r=audit_rhythm(case,d)
        self.assertEqual(r['missing_categories'],dict(removed_by_rhythm_audio=1))
