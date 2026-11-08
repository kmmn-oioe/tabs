import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tabgen.evaluate import evaluate_notes, evaluate_stages, run_case
from tabgen.tab import TabOptions


def note(start, pitch=65, part='lead', **kw):
    return dict(start=start,end=start+.2,pitch=pitch,part=part) | kw


def case(notes):
    return dict(evaluation_window=[0,2],notes=notes)


class EvaluationTests(unittest.TestCase):
    def test_one_to_one_matching_cannot_count_duplicate_predictions_twice(self):
        result = evaluate_notes(case([note(.5)]), [note(.5),note(.51)])
        self.assertEqual(result['overall']['matched'],1)
        self.assertEqual(result['overall']['extra'],1)
        self.assertEqual(result['overall']['missing'],0)
        self.assertEqual(result['overall']['precision'],.5)
        self.assertEqual(result['overall']['recall'],1)

    def test_repeated_attacks_use_maximum_matching_not_greedy(self):
        result = evaluate_notes(case([note(.1),note(.2)]), [note(.15),note(.05)],.06)
        self.assertEqual(result['overall']['matched'],2)

    def test_wrong_part_is_visible_without_counting_as_correct_part(self):
        result = evaluate_notes(case([note(.5)]), [note(.5,part='rhythm')])
        self.assertEqual(result['overall']['matched'],0)
        self.assertEqual(result['pitch_onset_ignoring_part']['matched'],1)
        self.assertEqual(len(result['part_confusions']),1)
        self.assertEqual(result['parts']['lead']['missing'],1)
        self.assertEqual(result['parts']['rhythm']['extra'],1)

    def test_unison_notes_in_two_parts_need_two_predictions(self):
        ref = [note(.5),note(.5,part='rhythm')]
        result = evaluate_notes(case(ref),[note(.5)])
        self.assertEqual(result['overall']['matched'],1)
        self.assertEqual(result['overall']['missing'],1)

    def test_tolerance_window_rests_and_alternative_fingering(self):
        c = case([note(.5,string=0,fret=1)])
        c['evaluation_window'] = [.2,1]
        c['rests'] = [dict(part='lead',start=.2,end=.4)]
        result = evaluate_notes(c,[note(.3),note(.56,string=1,fret=6),note(1.5)])
        self.assertEqual(result['overall']['matched'],1)
        self.assertEqual(result['overall']['extra'],1)
        self.assertEqual(result['overall']['same_fingering_rate'],0)
        self.assertAlmostEqual(result['overall']['onset_mae_ms'],60)
        self.assertEqual(result['annotated_rests'][0]['false_attacks'],1)
        self.assertEqual(evaluate_notes(c,[note(.59)])['overall']['matched'],0)

    def test_empty_data_does_not_claim_perfect_accuracy(self):
        result = evaluate_notes(case([]),[])
        self.assertIsNone(result['overall']['f1'])
        self.assertIsNone(result['overall']['precision'])
        self.assertEqual(evaluate_notes(case([note(.5)]),[])['overall']['f1'],0)

    def test_invalid_notes_and_tolerance_are_rejected(self):
        for tolerance in [0,-.1,math.nan]:
            with self.assertRaises(ValueError):
                evaluate_notes(case([]),[],tolerance)
        for n in [note(-1),note(.1,pitch=128),note(.1,end=.05),note(math.inf)]:
            with self.assertRaises(ValueError):
                evaluate_notes(case([n]),[])

    def test_actual_reference_self_scores_and_capo_pitch_is_consistent(self):
        path = Path(__file__).resolve().parents[1]/'benchmarks/cases/unmei_intro.json'
        reference = json.loads(path.read_text(encoding='utf-8'))
        result = evaluate_notes(reference,copy.deepcopy(reference['notes']))
        self.assertEqual(result['overall']['matched'],85)
        self.assertEqual(result['overall']['f1'],1)
        self.assertEqual(sum(n['part']=='lead' for n in reference['notes']),40)
        for n in reference['notes']:
            self.assertEqual(n['pitch'],reference['tuning'][n['string']]+reference['capo']+n['fret'])
        for rest in reference['rests']:
            self.assertFalse(any(rest['start'] <= n['start'] < rest['end'] for n in reference['notes']))

    def test_raw_candidates_are_scored_without_inventing_part_labels(self):
        c=case([note(.5)])
        result=evaluate_stages(c,dict(transcribed=[dict(start=.5,end=.7,pitch=65)],
                                     lead_selected=[],lead=[],rhythm=[]))
        self.assertEqual(result['raw_candidates']['pitch_onset_ignoring_part']['recall'],1)
        self.assertEqual(result['raw_candidates']['part_confusions'],[])
        self.assertEqual(result['arranged']['overall']['recall'],0)

    def test_audio_gate_and_tracker_can_be_evaluated_separately(self):
        c=case([note(.5)])
        detected=dict(start=.5,end=.7,pitch=65)
        result=evaluate_stages(c,dict(transcribed=[detected],audio_supported=[detected],
                                     lead_selected=[],lead=[],rhythm=[]))
        self.assertEqual(result['audio_supported']['parts']['lead']['matched'],1)
        self.assertEqual(result['lead_before_quantization']['parts']['lead']['matched'],0)

    def test_pipeline_diagnostics_do_not_change_results(self):
        from tabgen.pipeline import generate
        raw=[dict(start=.5,end=.9,pitch=p,confidence=.9) for p in [48,52,55]]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audio.wav';path.touch()
            with patch('tabgen.transcribe.transcribe',return_value=raw), \
                 patch('tabgen.harmony.audio_chord_changes',return_value=[]):
                plain=generate(str(path),TabOptions(audio_lead_detection=False),split_lead_rhythm=True,auto_bpm=False)
                diagnostics={}
                measured=generate(str(path),TabOptions(audio_lead_detection=False),split_lead_rhythm=True,
                                  auto_bpm=False,diagnostics=diagnostics)
            self.assertEqual(plain[0],measured[0])
            self.assertEqual(diagnostics['transcribed'],raw)
            self.assertFalse(diagnostics['separation_applied'])
            self.assertIn('lead_selected',diagnostics)
            self.assertIn('rhythm',diagnostics)
            diagnostics['transcribed'][0]['pitch']=1
            self.assertEqual(raw[0]['pitch'],48)

    def test_mismatched_audio_is_rejected_before_running_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio=Path(tmp)/'wrong.mp3';audio.write_bytes(b'wrong file')
            reference=Path(tmp)/'reference.json'
            reference.write_text(json.dumps(dict(audio_sha256='incorrect')))
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                run_case(reference,audio,Path(tmp)/'out')

    def test_full_context_changes_analysis_length_without_changing_scoring_window(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            audio=Path(tmp)/'audio.wav';audio.touch()
            reference=Path(tmp)/'reference.json'
            c=case([])|dict(id='test',bpm=120,grid_offset=0,analysis_window=[0,1.5],
                            evaluation_window=[.2,1.],tuning=[64,59,55,50,45,40])
            reference.write_text(json.dumps(c))
            lengths=[]
            def generate(path,opts,**kwargs):
                lengths.append(sf.info(path).duration)
                kwargs['diagnostics'].update(transcribed=[],lead_selected=[],lead=[],rhythm=[],
                                             separation_applied=False)
                return '', '', '', ''
            with patch('tabgen.audio.read_preview_audio',return_value=(np.zeros((20,2)),10)), \
                 patch('tabgen.pipeline.generate',side_effect=generate):
                excerpt=run_case(reference,audio,Path(tmp)/'excerpt')
                full=run_case(reference,audio,Path(tmp)/'full',full_context=True)
            self.assertEqual(lengths,[1.5,2.])
            self.assertEqual(full['evaluation_window'],excerpt['evaluation_window'])
            self.assertEqual(full['run']['context_mode'],'full')


if __name__ == '__main__':
    unittest.main()
