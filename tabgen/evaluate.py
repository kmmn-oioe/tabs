"""Offline, one-to-one note evaluation against manually annotated TAB excerpts.

Run with ``python -m tabgen.evaluate --help``. Ground truth is never used by the
production transcriber, splitter or arranger; reference tempo is an explicitly
labelled diagnostic mode, distinct from the automatic baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import tempfile
from dataclasses import asdict
from importlib import metadata
from pathlib import Path


PARTS = ('lead', 'rhythm')


def _validate_notes(notes):
    for n in notes:
        if (not all(math.isfinite(n[k]) for k in ('start', 'end', 'pitch'))
                or n['start'] < 0 or n['end'] <= n['start']
                or n['pitch'] != int(n['pitch']) or not 0 <= n['pitch'] <= 127):
            raise ValueError(f'Invalid note: {n}')
        if n.get('part') not in (*PARTS, 'unassigned'):
            raise ValueError('Each note needs part=lead, rhythm or unassigned')


def _match(reference, predicted, tolerance, same_part=True):
    """Maximum-cardinality matching, then smallest total onset error.

    A greedy nearest-note match can lose true positives on repeated attacks.
    Each reference and prediction can participate in at most one match.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not reference or not predicted:
        return []
    # One invalid edge must cost more than every possible valid edge combined.
    count = min(len(reference), len(predicted))
    invalid = (count + 1) * 10
    cost = np.full((len(reference), len(predicted)), float(invalid))
    for i, r in enumerate(reference):
        for j, p in enumerate(predicted):
            delta = abs(r['start'] - p['start'])
            if (r['pitch'] == p['pitch'] and delta <= tolerance + 1e-9
                    and (not same_part or r['part'] == p['part'])):
                cost[i, j] = delta / tolerance
                if not same_part and r['part'] != p['part']:
                    cost[i, j] += 1e-6  # Prefer same part when timings tie.
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < invalid]


def _stats(reference, predicted, pairs):
    tp = len(pairs)
    precision = tp / len(predicted) if predicted else None
    recall = tp / len(reference) if reference else None
    total = len(reference) + len(predicted)
    onset = [abs(predicted[j]['start'] - reference[i]['start']) * 1000 for i,j in pairs]
    offset = [abs(predicted[j]['end'] - reference[i]['end']) * 1000 for i,j in pairs]
    fingering = [(reference[i]['string'], reference[i]['fret']) ==
                 (predicted[j]['string'], predicted[j]['fret']) for i,j in pairs
                 if all(k in reference[i] and k in predicted[j] for k in ('string','fret'))]
    return dict(reference_notes=len(reference), predicted_notes=len(predicted),
                matched=tp, extra=len(predicted)-tp, missing=len(reference)-tp,
                precision=precision, recall=recall, f1=2*tp/total if total else None,
                onset_mae_ms=statistics.mean(onset) if onset else None,
                score_noteoff_mae_ms=statistics.mean(offset) if offset else None,
                same_fingering_rate=statistics.mean(fingering) if fingering else None)


def evaluate_notes(case, predicted, tolerance=.08):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Onset tolerance must be positive and finite')
    start, end = case['evaluation_window']
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError('Invalid evaluation window')
    _validate_notes(case['notes'])
    _validate_notes(predicted)
    ref = [n for n in case['notes'] if start <= n['start'] < end]
    pred = [n for n in predicted if start <= n['start'] < end]
    pairs = _match(ref, pred, tolerance)
    used_r, used_p = {i for i,j in pairs}, {j for i,j in pairs}
    agnostic = _match(ref, pred, tolerance, same_part=False)
    result = dict(onset_tolerance_ms=tolerance*1000,
                  overall=_stats(ref, pred, pairs),
                  pitch_onset_ignoring_part=_stats(ref, pred, agnostic), parts={})
    for part in PARTS:
        rs, ps = [n for n in ref if n['part'] == part], [n for n in pred if n['part'] == part]
        result['parts'][part] = _stats(rs, ps, _match(rs, ps, tolerance))
    result['part_confusions'] = [dict(reference_id=ref[i].get('id',i),
        reference_part=ref[i]['part'], predicted_part=pred[j]['part'], pitch=ref[i]['pitch'],
        reference_start=ref[i]['start'], predicted_start=pred[j]['start'])
        for i,j in agnostic if ref[i]['part'] != pred[j]['part'] and pred[j]['part'] != 'unassigned']
    result['missing'] = [n for i,n in enumerate(ref) if i not in used_r]
    result['extra'] = [n for j,n in enumerate(pred) if j not in used_p]
    result['matches'] = [dict(reference_id=ref[i].get('id',i), predicted_index=j,
                             onset_error_ms=(pred[j]['start']-ref[i]['start'])*1000)
                         for i,j in pairs]
    result['annotated_rests'] = [dict(part=r['part'], start=r['start'], end=r['end'],
        false_attacks=sum(p['part'] == r['part'] and r['start'] <= p['start'] < r['end'] for p in pred))
        for r in case.get('rests', [])]
    return result


def evaluate_stages(case, diagnostics, tolerance=.08):
    def part_notes(key, part):
        return [dict(n, part=part) for n in diagnostics.get(key, [])]
    arranged = part_notes('lead','lead') + part_notes('rhythm','rhythm')
    lead_case = dict(case, notes=[n for n in case['notes'] if n['part'] == 'lead'],
                     rests=[r for r in case.get('rests',[]) if r['part'] == 'lead'])
    result = dict(
        arranged=evaluate_notes(case, arranged, tolerance),
        lead_before_quantization=evaluate_notes(lead_case, part_notes('lead_selected','lead'), tolerance),
        raw_candidates=evaluate_notes(case, part_notes('transcribed','unassigned'), tolerance),
        raw_coverage_by_reference_part={part: evaluate_notes(
            dict(case, notes=[n for n in case['notes'] if n['part'] == part]),
            part_notes('transcribed','unassigned'), tolerance)['pitch_onset_ignoring_part']
            for part in PARTS},
        arranged_sensitivity={str(ms): evaluate_notes(case, arranged, ms/1000)['overall']
                              for ms in (50,80,100)})
    if diagnostics.get('audio_supported') is not None:
        result['audio_supported'] = evaluate_notes(lead_case, part_notes('audio_supported','lead'), tolerance)
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions():
    versions = {'python': platform.python_version()}
    for name in ('basic-pitch','numpy','scipy','tensorflow','demucs','torch'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def report_markdown(report):
    def pct(value):
        return '—' if value is None else f'{value*100:.1f}%'
    lines = [f"# {report['case_id']} 比較結果", '',
             f"モード: {report['run']['tempo_mode']} / 分離適用: {report['run']['separation_applied']}",
             'リズム出力: '+('検出した発音を保持' if report['run']['options'].get('rhythm_mode') == 'notes'
                            else 'コード伴奏（保持）'),
             f"出力BPM: {report['run']['options']['bpm']:g} / TAB記載BPM: {report['reference_bpm']:g}",
             f"発音許容差: ±{report['onset_tolerance_ms']:g} ms。評価は指定した短い区間に限定。", '',
             '| 段階・パート | 正解音 | 出力音 | 一致 | 余計な音 | 抜けた音 | 適合率 | 再現率 | F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    scores = report['scores']
    rows = [('音程候補・パート不問', scores['raw_candidates']['pitch_onset_ignoring_part']),
            ('量子化前リード', scores['lead_before_quantization']['parts']['lead']),
            ('最終リード', scores['arranged']['parts']['lead']),
            ('最終リズム', scores['arranged']['parts']['rhythm']),
            ('最終出力全体', scores['arranged']['overall'])]
    if 'audio_supported' in scores:
        rows.insert(1, ('音声根拠あり・追跡前', scores['audio_supported']['parts']['lead']))
    for name, s in rows:
        lines.append(f"| {name} | {s['reference_notes']} | {s['predicted_notes']} | {s['matched']} | {s['extra']} | {s['missing']} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} |")
    lines += ['', f"パート不問で一致した音のうち役割が異なる対応: {len(scores['arranged']['part_confusions'])} 件。", '',
              '音程候補で見つかっていた正解音の割合（パート判定前）: '
              + ' / '.join(f"{p}: {pct(scores['raw_coverage_by_reference_part'][p]['recall'])}" for p in PARTS), '',
              '注意:', '',
              '- 正解はTAB動画から手で転記した冒頭5小節。原曲の全ギタートラックを保証するものではない。',
              '- 生の音程候補には歌・ベースなども含まれる。「余計な音」をすべてギター誤検出とはみなせない。',
              ('- リズム出力は検出した発音を保持。検出・声部分けの誤りも残る。運指制限と量子化による省略はありえる。'
               if report['run']['options'].get('rhythm_mode') == 'notes'
               else '- リズム出力はコードを保持する伴奏への編曲。原譜のアルペジオとの相違も欠落・追加に含む。'),
              '- 音の終了時刻は譜面上の音価。実録音の残響・サステインの正解ではない。',
              '- 運指の一致は参考値。同じ音を別の弦で弾くことも可能。',
              '- referenceモードは正解のBPM・グリッドを与えた診断。自動採譜の精度として扱わない。',
              '- 許容差50/80/100 msの感度と、個々の不足・追加音はJSONレポートに記録。', '']
    return '\n'.join(lines)


def run_case(reference_path, audio_path, output_dir, *, tempo_mode='auto', separation=False, tolerance=.08,
             full_context=False, rhythm_mode='chords'):
    import soundfile as sf
    from .audio import read_preview_audio
    from .pipeline import generate
    from .tab import TabOptions

    case = json.loads(Path(reference_path).read_text(encoding='utf-8'))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Onset tolerance must be positive and finite')
    audio_hash = _sha(audio_path)
    if case.get('audio_sha256') and audio_hash != case['audio_sha256']:
        raise ValueError('原曲ファイルのSHA-256が正解データと異なります。同じ添付MP3を指定してください。')
    if rhythm_mode not in ('notes','chords'):
        raise ValueError('Unknown rhythm mode')
    opts = TabOptions(capo=case.get('capo',0), tuning=case['tuning'], rhythm_mode=rhythm_mode)
    if tempo_mode == 'reference':
        opts.bpm, opts.grid_offset = case['bpm'], case['grid_offset']
    elif tempo_mode != 'auto':
        raise ValueError('Unknown tempo mode')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audio, sr = read_preview_audio(str(audio_path))
    begin, stop = case['analysis_window']
    if full_context:
        stop = len(audio)/sr
    if begin != 0 or len(audio)/sr < stop:
        raise ValueError('Analysis must start at zero and fit within the supplied audio')
    if case['evaluation_window'][1] > stop:
        raise ValueError('Evaluation window must fit within the analyzed audio')
    diagnostics = {}
    with tempfile.TemporaryDirectory(prefix='tabgen_evaluate_') as tmp:
        wav = Path(tmp)/'excerpt.wav'
        sf.write(wav, audio[:round(stop*sr)], sr, subtype='FLOAT')
        text, _, _, _ = generate(str(wav), opts, split_lead_rhythm=True,
            use_separation=separation, auto_bpm=tempo_mode=='auto', diagnostics=diagnostics,
            progress=lambda pct,msg: print(msg, flush=True))
    report = dict(case_id=case['id'], reference_bpm=case['bpm'],
        evaluation_window=case['evaluation_window'], analysis_window=[begin,stop],
        onset_tolerance_ms=tolerance*1000,
        run=dict(tempo_mode=tempo_mode, context_mode='full' if full_context else 'excerpt',
                 tempo_estimation=diagnostics.get('tempo'),
                 lead_evidence_method=diagnostics.get('lead_evidence_method'), separation_requested=separation,
                 separation_applied=diagnostics['separation_applied'], options=asdict(opts),
                 audio_sha256=audio_hash, reference_sha256=_sha(reference_path),
                 source_sha256=_source_hash(), versions=_versions()),
        scores=evaluate_stages(case, diagnostics, tolerance))
    (output/'diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.md').write_text(report_markdown(report),encoding='utf-8')
    (output/'tab.txt').write_text(text,encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='正解TABと現在のアプリ出力を区間比較します')
    sub = parser.add_subparsers(dest='command',required=True)
    run = sub.add_parser('run')
    run.add_argument('--reference', required=True)
    run.add_argument('--audio', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--tempo-mode', choices=('auto','reference'),default='auto')
    run.add_argument('--separation',action='store_true')
    run.add_argument('--full-context',action='store_true',help='全曲を解析し、同じ注釈区間を採点する')
    run.add_argument('--rhythm-mode',choices=('chords','notes'),default='chords')
    run.add_argument('--tolerance-ms',type=float,default=80)
    compare = sub.add_parser('compare')
    compare.add_argument('--reference',required=True)
    compare.add_argument('--diagnostics',required=True)
    compare.add_argument('--output',required=True)
    compare.add_argument('--tolerance-ms',type=float,default=80)
    args = parser.parse_args()
    if args.command == 'run':
        result = run_case(args.reference,args.audio,args.output,tempo_mode=args.tempo_mode,
                          separation=args.separation,tolerance=args.tolerance_ms/1000,
                          full_context=args.full_context,rhythm_mode=args.rhythm_mode)
        print(report_markdown(result))
    else:
        case = json.loads(Path(args.reference).read_text(encoding='utf-8'))
        diagnostics = json.loads(Path(args.diagnostics).read_text(encoding='utf-8'))
        result = evaluate_stages(case,diagnostics,args.tolerance_ms/1000)
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
