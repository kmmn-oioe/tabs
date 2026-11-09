from dataclasses import asdict
import unittest

from benchmarks.audit_rhythm import audit_rhythm
from tabgen.rhythm import arrange_rhythm
from tabgen.tab import TabOptions


class RhythmAuditTests(unittest.TestCase):
    def note(self, pitch, start, part=None, identifier=None):
        n = dict(pitch=pitch, start=start, end=start+.1, confidence=.9)
        if part:
            n.update(part=part, id=identifier)
        return n

    def diagnostics(self, raw, candidates=None):
        opts = TabOptions(bpm=120, rhythm_mode='notes')
        candidates = list(raw) if candidates is None else candidates
        arranged, _ = arrange_rhythm(candidates, opts, changes=[])
        return dict(transcribed=raw, rhythm_candidates=candidates, rhythm=arranged,
                    options=asdict(opts), lead_audio_scores=[0.]*len(raw))

    def test_quantization_and_wrong_part_are_distinct_from_unmatched_detection(self):
        raw = [self.note(64, .07), self.note(67, .25), self.note(60, .75)]
        case = dict(id='fixture', evaluation_window=[0., 1.], notes=[
            self.note(64, 0., 'rhythm', 'rhythm.0'),
            self.note(67, .25, 'lead', 'lead.0')])
        result = audit_rhythm(case, self.diagnostics(raw))
        self.assertEqual(result['extra_categories'], dict(quantized_out_of_tolerance=1,
            lead_reference_pitch_onset=1, unmatched_before_and_after=1))
        self.assertEqual(result['missing_categories'], dict(quantized_out_of_tolerance=1))
        self.assertEqual([n['raw_index'] for n in result['extras']], [0, 1, 2])
        self.assertEqual(result['stages']['transcribed']['matched'], 1)
        self.assertEqual(result['stages']['rhythm']['matched'], 0)

    def test_shared_unison_removed_by_split_is_not_called_missing_transcription(self):
        raw = [self.note(65, 0.)]
        case = dict(id='fixture', evaluation_window=[0., 1.], notes=[
            self.note(65, 0., 'rhythm', 'rhythm.0'), self.note(65, 0., 'lead', 'lead.0')])
        result = audit_rhythm(case, self.diagnostics(raw, []))
        self.assertEqual(result['missing_categories'], dict(removed_by_lead_selection=1))
        self.assertEqual(result['missing'][0]['shared_with_lead_reference_ids'], ['lead.0'])

    def test_unreproducible_or_untraceable_output_is_rejected(self):
        raw = [self.note(64, 0.)]
        case = dict(id='fixture', evaluation_window=[0., 1.], notes=[])
        d = self.diagnostics(raw)
        d['rhythm'][0]['pitch'] = 65
        with self.assertRaisesRegex(ValueError, 'Replay differs'):
            audit_rhythm(case, d)
        d = self.diagnostics(raw)
        d['rhythm_candidates'][0] = self.note(67, .25)
        with self.assertRaisesRegex(ValueError, 'no original'):
            audit_rhythm(case, d)

    def test_lead_match_before_rounding_is_not_reported_as_unknown_detection(self):
        case = dict(id='fixture', evaluation_window=[0., 1.], notes=[
            self.note(72, 0., 'lead', 'lead.0')])
        result = audit_rhythm(case, self.diagnostics([self.note(72, .07)]))
        self.assertEqual(result['extra_categories'], dict(lead_reference_pitch_onset=1))
        self.assertEqual(result['extras'][0]['raw_lead_reference_ids'], ['lead.0'])
        self.assertEqual(result['extras'][0]['final_lead_reference_ids'], [])

    def test_chord_arrangement_is_not_misreported_as_preserved_detections(self):
        d = self.diagnostics([])
        d['options']['rhythm_mode'] = 'chords'
        with self.assertRaisesRegex(ValueError, 'rhythm_mode=notes'):
            audit_rhythm(dict(id='fixture', evaluation_window=[0., 1.], notes=[]), d)
