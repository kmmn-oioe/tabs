"""Replay part analysis with private fixed inputs, without model inference.

Run from a Windows or Linux checkout after installing the core requirements.
This is a regression check: separation and pitch detections are reused, while
audio evidence, role analysis, arrangement and exports actually run again.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'local-input')
    parser.add_argument('--output', type=Path, default=ROOT/'benchmark-output'/'handoff-replay')
    parser.add_argument('--condition', choices=('both', 'separated', 'unseparated'), default='both')
    parser.add_argument('--allow-changed', action='store_true',
                        help='Save changed results after development without treating them as an exact baseline replay')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    manifest_path = data/'manifest.json'
    if not manifest_path.is_file():
        parser.error('local-input/manifest.json is missing. Extract the private handoff data first.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = (data/entry['path']).resolve()
        if not path.is_relative_to(data) or not path.is_file():
            parser.error('Missing or unsafe data path: '+entry['path'])
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            parser.error('Private data SHA-256 mismatch: '+entry['path'])
    reference = ROOT/'benchmarks'/'cases'/'unmei_intro.json'
    if digest(reference) != manifest['reference_sha256']:
        parser.error('The reference annotation differs from the handoff snapshot.')
    from tabgen.evaluate import run_case
    conditions = ('separated', 'unseparated') if args.condition == 'both' else (args.condition,)
    changed = []
    for name in conditions:
        baseline_dir = data/'baseline'/name
        cached = json.loads((baseline_dir/'diagnostics.json').read_text(encoding='utf-8'))
        baseline = json.loads((baseline_dir/'report.json').read_text(encoding='utf-8'))
        with ExitStack() as stack:
            stack.enter_context(patch('tabgen.transcribe.transcribe', return_value=cached['transcribed']))
            if name == 'separated':
                stack.enter_context(patch('tabgen.separate.demucs_available', return_value=True))
                stack.enter_context(patch('tabgen.separate.separate_guitar', return_value=str(data/'fixed'/'guitar.wav')))
            result = run_case(reference, data/'media'/'original.mp3', args.output/name,
                              tempo_mode='reference', rhythm_mode='notes', separation=name == 'separated')
        result['run']['handoff_replay'] = dict(
            separation_model_executed=False, pitch_model_executed=False,
            audio_evidence_and_arrangement_executed=True,
            input_manifest_sha256=digest(manifest_path),
            explanation='Fixed pitch detections and retained separation reused. This is not fresh model accuracy.')
        if name == 'separated':
            import soundfile as sf
            info = sf.info(str(data/'fixed'/'guitar.wav'))
            result['run']['separation_details'] = dict(
                manifest['fixed_separation'], execution='Retained output reused; Demucs not executed',
                output_sha256=digest(data/'fixed'/'guitar.wav'),
                frames=info.frames, samplerate=info.samplerate, channels=info.channels)
        (args.output/name/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        markdown = args.output/name/'report.md'
        markdown.write_text('> 固定入力リプレイです。分離・音程検出モデルは実行せず、保存済み入力からパート判定と整形を再実行しました。\n\n'
                            + markdown.read_text(encoding='utf-8'), encoding='utf-8')
        for part in ('lead', 'rhythm'):
            actual = result['scores']['arranged']['parts'][part]
            expected = baseline['scores']['arranged']['parts'][part]
            counts = {key:actual[key] for key in ('matched', 'extra', 'missing')}
            print(name, part, counts)
            if any(actual[key] != expected[key] for key in counts):
                changed.append(name+'/'+part)
    if changed and not args.allow_changed:
        raise SystemExit('Baseline differs for '+', '.join(changed)+
                         '. Inspect reports; use --allow-changed only to compare intentional development changes.')
    print('Fixed-input replay complete. Reports: '+str(args.output.resolve()))


if __name__ == '__main__':
    main()
