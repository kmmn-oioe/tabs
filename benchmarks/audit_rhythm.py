"""Trace rhythm extras back to cached detections without running audio models.

Usage: python -m benchmarks.audit_rhythm --reference CASE --diagnostics JSON --output JSON
Ground truth is used for offline diagnosis only. Production settings are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from tabgen.evaluate import evaluate_notes
from tabgen.rhythm import arrange_rhythm
from tabgen.tab import TabOptions


def audit_rhythm(case, diagnostics, tolerance=.08):
    opts = TabOptions(**diagnostics['options'])
    if opts.rhythm_mode != 'notes':
        raise ValueError('This audit requires rhythm_mode=notes')
    raw = diagnostics['transcribed']
    available = defaultdict(deque)
    for i, note in enumerate(raw):
        available[json.dumps(note, sort_keys=True)].append(i)
    tagged = []
    for note in diagnostics.get('rhythm_arrangement_input', diagnostics['rhythm_candidates']):
        original = {k:v for k,v in note.items() if k not in ('_snap_start', '_audio_rejection')}
        indices = available[json.dumps(original, sort_keys=True)]
        if not indices:
            raise ValueError('A rhythm candidate has no original transcription')
        tagged.append(dict(note, _audit_raw_index=indices.popleft()))
    replay, _ = arrange_rhythm(tagged, opts, changes=[])
    role_removed = diagnostics.get('rhythm_role_removed', [])
    without_tag = lambda n: {k:v for k,v in n.items() if k != '_audit_raw_index'}
    for note in role_removed:
        proofs = [p for p in diagnostics.get('part_role_checks', [])
                  if p.get('accepted') and note in p.get('removed_rhythm', [])]
        if len(proofs) != 1 or sum(without_tag(n) == note for n in replay) != 1:
            raise ValueError('A role removal has no unique proof or original rhythm note')
        replay = [n for n in replay if without_tag(n) != note]
    restored = diagnostics.get('rhythm_unison_restored', [])
    for note in restored:
        proofs = [p for p in diagnostics.get('unison_checks', [])
                  if p.get('restored') and p.get('restored_note') == note]
        if len(proofs) != 1:
            raise ValueError('A restored unison has no unique proof')
        i = proofs[0]['raw_index']
        if (not isinstance(i, int) or not 0 <= i < len(raw)
                or raw[i]['pitch'] != note['pitch'] or raw[i]['confidence'] != note['confidence']):
            raise ValueError('A restored unison has no matching raw detection')
        replay.append(dict(note, _audit_raw_index=i))
    if restored:
        replay.sort(key=lambda n:n['start'])
    plain = [{k: v for k, v in n.items() if k != '_audit_raw_index'} for n in replay]
    if plain != diagnostics['rhythm']:
        raise ValueError('Replay differs from saved rhythm; do not infer provenance')

    lo, hi = case['evaluation_window']
    reference = [n for n in case['notes'] if lo <= n['start'] < hi]
    def near(note, part, octaves=False):
        return [n for n in reference if n['part'] == part
                and abs(n['start']-note['start']) <= tolerance+1e-9
                and ((n['pitch'] != note['pitch'] and (n['pitch']-note['pitch']) % 12 == 0)
                     if octaves else n['pitch'] == note['pitch'])]
    def score(notes):
        return evaluate_notes(case, [dict(n, part='rhythm') for n in notes], tolerance)
    stages = {k: score(diagnostics[k]) for k in ('transcribed', 'rhythm_candidates', 'rhythm')}
    if 'rhythm_audio_supported' in diagnostics:
        stages['rhythm_audio_supported'] = score(diagnostics['rhythm_audio_supported'])
    final_window = [(i, n) for i, n in enumerate(replay) if lo <= n['start'] < hi]
    matched = {m['predicted_index'] for m in stages['rhythm']['matches']}
    extras = []
    for window_index, (output_index, note) in enumerate(final_window):
        if window_index in matched:
            continue
        index = note['_audit_raw_index']
        original = raw[index]
        raw_matches = near(original, 'rhythm')
        final_matches = near(note, 'rhythm')
        lead_matches = near(note, 'lead')
        raw_lead_matches = near(original, 'lead')
        category = ('competing_rhythm_detection' if final_matches else
                    'quantized_out_of_tolerance' if raw_matches else
                    'lead_reference_pitch_onset' if lead_matches or raw_lead_matches else
                    'unmatched_before_and_after')
        extras.append(dict(output_index=output_index, raw_index=index,
            detected=original, arranged=plain[output_index], category=category,
            raw_rhythm_reference_ids=[n['id'] for n in raw_matches],
            raw_lead_reference_ids=[n['id'] for n in raw_lead_matches],
            final_lead_reference_ids=[n['id'] for n in lead_matches],
            octave_related_reference_ids=[n['id'] for part in ('lead', 'rhythm')
                                          for n in near(note, part, octaves=True)],
            lead_audio_score=(diagnostics['lead_audio_scores'][index]
                              if diagnostics.get('lead_audio_scores') is not None else None)))

    ref_by_id = {n['id']: n for n in reference}
    matched_ids = {key: {m['reference_id'] for m in value['matches']}
                   for key, value in stages.items()}
    missing = []
    for note in stages['rhythm']['missing']:
        if note['part'] != 'rhythm':
            continue
        ref_id = note['id']
        category = ('absent_from_transcription' if ref_id not in matched_ids['transcribed'] else
                    'removed_by_lead_selection' if ref_id not in matched_ids['rhythm_candidates'] else
                    'removed_by_rhythm_audio' if 'rhythm_audio_supported' in matched_ids
                    and ref_id not in matched_ids['rhythm_audio_supported'] else
                    'quantized_out_of_tolerance' if any(
                        ref_id in x['raw_rhythm_reference_ids'] for x in extras
                        if x['category'] == 'quantized_out_of_tolerance') else
                    'dropped_during_arrangement')
        missing.append(dict(reference=note, category=category,
            shared_with_lead_reference_ids=[n['id'] for n in near(ref_by_id[ref_id], 'lead')]))
    sweep = []
    for confidence in (.3, .35, .4, .45, .5, .55):
        adjusted, _ = arrange_rhythm(tagged, replace(opts, min_confidence=confidence), changes=[])
        adjusted = [n for n in adjusted if without_tag(n) not in role_removed]
        # Keep the saved audio/unison decisions fixed, as with attack filtering
        # and grid calibration; only the arrangement threshold is swept.
        adjusted += [dict(n) for n in restored if n['confidence'] >= confidence]
        sweep.append(dict(min_confidence=confidence, **score(adjusted)['parts']['rhythm']))
    return dict(case_id=case['id'], evaluation_window=case['evaluation_window'],
        options=diagnostics['options'], onset_tolerance_ms=tolerance*1000,
        provenance_replay_identical=True,
        stages={k: v['parts']['rhythm'] for k, v in stages.items()},
        extra_categories=dict(Counter(n['category'] for n in extras)),
        missing_categories=dict(Counter(n['category'] for n in missing)),
        extras=extras, missing=missing, confidence_sweep=sweep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--diagnostics', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit_rhythm(json.loads(args.reference.read_text(encoding='utf-8')),
                          json.loads(args.diagnostics.read_text(encoding='utf-8')))
    result['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in [('reference', args.reference),
                                                ('diagnostics', args.diagnostics)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('stages', 'extra_categories', 'missing_categories')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
