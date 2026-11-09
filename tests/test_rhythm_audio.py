from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_fixtures import guitar_scene, SAMPLE_RATE
from tabgen.rhythm_audio import (analyze_rhythm_attacks, filter_rhythm_attacks,
                                  align_rhythm_grid)
from tabgen.tab import TabOptions, arrange_notes
from benchmarks.audit_rhythm import audit_rhythm


class RhythmAudioTests(unittest.TestCase):
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
