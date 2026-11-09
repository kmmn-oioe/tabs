"""Pipeline: audio/YouTube -> tab text + MIDI."""
from __future__ import annotations

import os
import tempfile
from typing import Callable

from .audio import load_audio, download_youtube
from .tab import TabOptions, notes_to_tab, arrange_notes, arrange_melody


def generate(
    source: str,
    opts: TabOptions | None = None,
    use_separation: bool = False,
    split_lead_rhythm: bool = False,
    lead_threshold: int = 64,
    min_note_length_ms: float = 70.0,
    auto_bpm: bool = True,
    progress: Callable[[str], None] | None = None,
    audio_sources: dict[str, str] | None = None,
    diagnostics: dict | None = None,
    prepare_playback: bool = False,
) -> tuple[str, str, str, str]:
    """Returns (tab_text, midi_paths, sheet_paths, stem_wav_path).

    source: local audio file path or YouTube URL.
    audio_sources: output paths (original/separated/backing/playback), and
        playback_error when guitar-free playback cannot be prepared.
    diagnostics: optional snapshots for offline evaluation; does not affect generation.
    prepare_playback: render guitar-free backing + final TAB synth for the GUI.
    """
    opts = opts or TabOptions()
    if opts.rhythm_mode not in ("notes", "chords"):
        raise ValueError("Unknown rhythm mode: " + opts.rhythm_mode)
    if audio_sources is not None:
        audio_sources.clear()
    if diagnostics is not None:
        diagnostics.clear()

    def log(msg: str, pct: float | None = None):
        if progress:
            progress(pct, msg) if pct is not None else progress(None, msg)

    tmp_dir = tempfile.mkdtemp(prefix="tabgen_")

    # 1. acquire audio
    if source.lower().startswith(("http://", "https://", "www.", "youtube.com", "youtu.be")):
        log("YouTubeから音声をダウンロード中…", 0)
        wav = download_youtube(source, tmp_dir, progress=progress)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"ファイルが見つかりません: {source}")
        wav = source

    # 2. optional stem separation
    if audio_sources is not None:
        audio_sources["original"] = wav
    stem_path = wav
    backing_path = None
    playback_error = ""
    if use_separation or prepare_playback:
        log("ギターパートを分離中…(Demucs・時間がかかります)")
        from .separate import demucs_available, separate_guitar
        if not demucs_available():
            log("※ demucs/torch が未インストールのため分離をスキップします")
            playback_error = "再生にはDemucs・torchとギター専用モデルが必要です。導入後にTABを再生成してください。"
        else:
            try:
                separated = separate_guitar(wav, tmp_dir, progress=progress)
            except (OSError, RuntimeError) as error:
                if use_separation:
                    raise
                playback_error = f"ギターオフ音源を作成できませんでした: {error}"
                log(playback_error)
            else:
                if use_separation:
                    stem_path = separated
                if prepare_playback:
                    from .separate import guitar_backing_path
                    try:
                        backing_path = guitar_backing_path(separated)
                    except RuntimeError as error:
                        playback_error = str(error)
    if audio_sources is not None and os.path.realpath(stem_path) != os.path.realpath(wav):
        audio_sources["separated"] = stem_path

    # 3. transcribe
    log("音程を解析中…", 50)
    from .transcribe import transcribe
    notes = transcribe(stem_path, progress=progress,
                       min_note_length_ms=min_note_length_ms,
                       stereo_candidates=split_lead_rhythm and opts.audio_lead_detection)
    log(f"{len(notes)} ノートを検出しました", 95)
    if diagnostics is not None:
        diagnostics["transcribed"] = [dict(n) for n in notes]

    # 3.5 tempo & grid-offset estimation
    if auto_bpm:
        log("テンポを推定中…", 96)
        from .tempo import estimate_bpm_and_offset
        try:
            tempo_diagnostics = {}
            bpm, offset = estimate_bpm_and_offset(stem_path, diagnostics=tempo_diagnostics)
            opts.bpm = round(bpm, 1)
            opts.grid_offset = round(offset, 3)
            if diagnostics is not None:
                diagnostics["tempo"] = tempo_diagnostics
            if tempo_diagnostics.get("fallback"):
                log(f"拍の情報が不足しているため BPM {opts.bpm:g} を仮設定しました", 96)
            else:
                log(f"BPM {opts.bpm:g} を自動推定しました", 96)
        except Exception as e:
            log(f"テンポ推定をスキップ: {e}")

    # 4. build tab
    log("タブ譜を組み立て中…", 96)
    duration = max((n["end"] for n in notes), default=0.)

    def with_header(text: str) -> str:
        return f"BPM {opts.bpm:g}  [{opts.difficulty}]" + (f"  (カポ {opts.capo})" if opts.capo else "") + "\n" + text

    def arrange(part: list[dict]) -> list[dict]:
        arranged = arrange_notes(part, opts)
        log(f"演奏用に調整 ({opts.difficulty}): {len(part)} → {len(arranged)} ノート", 97)
        return arranged

    if split_lead_rhythm:
        from .rhythm import arrange_rhythm, progression_text
        from .split import split_notes
        lead_scores = None
        lead_evidence = None
        lead_evidence_method = "disabled"
        lead_method = "リード判定: 検出ノートのみ（音声による存在判定は無効）"
        if opts.audio_lead_detection:
            from .lead_audio import analyze_lead_audio
            try:
                log("音声の定位・倍音・発音から独立したリードの根拠を解析中…", 96)
                evidence = analyze_lead_audio(stem_path, notes, min_confidence=opts.min_confidence)
                lead_evidence = evidence
                lead_scores = evidence.scores
                lead_evidence_method = evidence.method
                lead_method = evidence.summary()
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                lead_scores = [0.] * len(notes)
                lead_evidence_method = "failed"
                lead_method = "リード音声判定に失敗したため保留（休符）。音声入力や依存関係を確認してください。"
                log(f"{lead_method}: {error}", 96)
        if diagnostics is not None:
            diagnostics["lead_audio_scores"] = lead_scores
            diagnostics["lead_evidence_method"] = lead_evidence_method
            diagnostics["audio_supported"] = ([dict(n) for n,s in zip(notes, lead_scores) if s >= .6]
                                               if lead_scores is not None else None)
        lead, rhythm = split_notes(notes, lead_threshold, min_confidence=opts.min_confidence,
                                   suppress_chords=opts.suppress_chord_attacks,
                                   start_seconds=opts.lead_start_seconds, lead_scores=lead_scores)
        log(f"旋律追跡: {len(lead)} ノートを採用 / その他 {len(rhythm)} ノート")
        if diagnostics is not None:
            diagnostics["lead_selected"] = [dict(n) for n in lead]
            diagnostics["rhythm_candidates"] = [dict(n) for n in rhythm]
        lead_voice_anchors = lead
        lead = arrange_melody(lead, opts)
        entry_text = (f"リード開始指定: {opts.lead_start_seconds:g}秒（それ以前は休符）\n"
                      if opts.lead_start_seconds > 0 else "")
        if lead:
            lead_text = notes_to_tab(lead, opts, arranged=True, duration=duration)
        else:
            lead_text = "（リードを確認できず／休符）" if opts.audio_lead_detection else "（リードなし／休符）"
        from .harmony import audio_chord_changes
        try:
            log("音声のクロマ特徴からコードと切替位置を解析中…", 97)
            changes = audio_chord_changes(stem_path, opts)
            analysis_method = "コード推定: 音声クロマ（拍単位・前後関係で平滑化）"
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            log(f"音声コード解析を利用できないため検出ノートから推定します: {error}", 97)
            changes = None
            analysis_method = "コード推定: 検出ノート（音声解析を利用できなかったため）"
        if changes is None:
            from .rhythm import chord_changes
            changes = chord_changes(notes, opts)
        rhythm_input = rhythm if opts.rhythm_mode == "notes" else notes
        rhythm_audio_method = ""
        if opts.rhythm_mode == "notes":
            from .rhythm_audio import analyze_rhythm_attacks, filter_rhythm_attacks, align_rhythm_grid
            try:
                log("リズム候補の音高別の発音を確認中…", 97)
                attacks = analyze_rhythm_attacks(stem_path, rhythm_input)
                supported = filter_rhythm_attacks(rhythm_input, attacks)
                voice_checks = []
                recovered = []
                if lead_evidence is not None:
                    from .rhythm_audio import rhythm_voice_checks, recover_lead_holes
                    key = lambda n: (n['pitch'], n['start'], n['confidence'])
                    by_note = {key(n): f for n, f in zip(notes, lead_evidence.features)}
                    voice_checks = rhythm_voice_checks(
                        supported, [by_note[key(n)] for n in supported],
                        lead_voice_anchors, [by_note[key(n)] for n in lead_voice_anchors])
                    supported = [dict(n, _audio_rejection=c['reason']) if c['reject'] else n
                                 for n, c in zip(supported, voice_checks)]
                    lead, recovered = recover_lead_holes(lead, supported, voice_checks, opts)
                    if recovered:
                        lead_text = notes_to_tab(lead, opts, arranged=True, duration=duration)
                        log(f"声部と発音を確認してリードの空き位置へ移動: {len(recovered)}音", 97)
                    if diagnostics is not None:
                        diagnostics['lead_recovered_from_rhythm'] = [dict(n) for n in recovered]
                rhythm_input, timing = align_rhythm_grid(supported, opts)
                rhythm_audio_method = f"リズム音声確認: {len(rhythm)} → {len(supported)}候補（基本音・発音の立ち上がり）\n"
                withheld = sum(c['reject'] for c in voice_checks)
                if withheld:
                    rhythm_audio_method += f"声部確認: リード声部に対応する残存候補{withheld}音をリズム出力から保留\n"
                if recovered:
                    rhythm_audio_method += f"発音も確認できた{len(recovered)}音をリードの空き位置へ移動\n"
                if timing['applied']:
                    rhythm_audio_method += f"拍位置調整: 検出の遅れを約{timing['delay_seconds']*1000:.0f}ms補正\n"
                if diagnostics is not None:
                    diagnostics.update(rhythm_attacks=attacks,
                                       rhythm_audio_supported=[dict(n) for n in supported if not n.get('_audio_rejection')],
                                       rhythm_voice_checks=voice_checks,
                                       rhythm_timing=timing)
            except (OSError, RuntimeError, ValueError, ImportError) as error:
                rhythm_input = rhythm
                rhythm_audio_method = "リズム音声確認: 利用できないため検出候補を保持\n"
                log(f"{rhythm_audio_method.strip()}: {error}", 97)
            if diagnostics is not None:
                diagnostics['rhythm_arrangement_input'] = [dict(n) for n in rhythm_input]
        rhythm, changes = arrange_rhythm(rhythm_input, opts, changes=changes)
        if opts.rhythm_mode == 'notes' and lead_evidence is not None:
            from .unison import recover_unisons
            try:
                log("伴奏の反復と同音の空間成分からユニゾンを確認中…", 97)
                rhythm, unisons, unison_checks = recover_unisons(
                    stem_path, lead, rhythm, notes, lead_evidence.features, opts)
                if unisons:
                    rhythm_audio_method += f"ユニゾン確認: 両声部の根拠がある{len(unisons)}音をリズムにも復元\n"
                if diagnostics is not None:
                    diagnostics.update(rhythm_unison_restored=unisons, unison_checks=unison_checks)
            except (OSError, RuntimeError, ValueError) as error:
                log(f"ユニゾン確認をスキップ（既存の発音を保持）: {error}", 97)
        rhythm_method = ("リズム出力: 検出した発音を保持（拍位置・運指を調整）" if opts.rhythm_mode == "notes"
                         else "リズム出力: コード伴奏（次の切替まで保持）")
        log(f"{rhythm_method}: {len(rhythm)} 音", 97)
        tab_text = (
            "====== リードギター ======\n"
            + lead_method + "\n"
            + entry_text + with_header(lead_text)
            + "\n\n====== リズムギター ======\n"
            + rhythm_method + "\n"
            + rhythm_audio_method
            + analysis_method + "\n"
            + progression_text(changes, opts) + "\n\n"
            + with_header(notes_to_tab(rhythm, opts, arranged=True,
                                       duration=duration if opts.rhythm_mode == "notes" else None))
        )
        lead_notes, rhythm_notes = lead, rhythm
    else:
        from .rhythm import chord_changes
        changes = chord_changes(notes, opts)
        notes = arrange(notes)
        tab_text = with_header(notes_to_tab(notes, opts, arranged=True))
        lead_notes, rhythm_notes = notes, []

    if diagnostics is not None:
        from dataclasses import asdict
        diagnostics.update(lead=[dict(n) for n in lead_notes],
                           rhythm=[dict(n) for n in rhythm_notes],
                           options=asdict(opts),
                           separation_applied=os.path.realpath(stem_path) != os.path.realpath(wav))

    # 5. export MIDI + sheet music (MusicXML)
    midi_paths: list[str] = []
    sheet_paths: list[str] = []
    try:
        from .export import export_midi
        if split_lead_rhythm:
            export_midi(lead_notes, os.path.join(tmp_dir, "tabgen_lead.mid"))
            export_midi(rhythm_notes, os.path.join(tmp_dir, "tabgen_rhythm.mid"))
            midi_paths = [os.path.join(tmp_dir, "tabgen_lead.mid"), os.path.join(tmp_dir, "tabgen_rhythm.mid")]
        else:
            export_midi(notes, os.path.join(tmp_dir, "tabgen_output.mid"))
            midi_paths = [os.path.join(tmp_dir, "tabgen_output.mid")]
    except Exception as e:
        log(f"MIDI出力をスキップ: {e}")

    try:
        from .sheet import export_musicxml
        from .tabpdf import export_tab_data
        log("楽譜(MusicXML)を作成中…", 98)
        if split_lead_rhythm:
            export_musicxml(lead_notes, os.path.join(tmp_dir, "tabgen_lead.musicxml"), opts, "lead",
                            duration=duration, start_floor=opts.lead_start_seconds)
            export_musicxml(rhythm_notes, os.path.join(tmp_dir, "tabgen_rhythm.musicxml"), opts, "rhythm",
                            duration=duration if opts.rhythm_mode == "notes" else None,
                            chord_changes=changes if opts.rhythm_mode == "notes" else None)
            sheet_paths = [os.path.join(tmp_dir, "tabgen_lead.musicxml"),
                           os.path.join(tmp_dir, "tabgen_rhythm.musicxml")]
        else:
            export_musicxml(notes, os.path.join(tmp_dir, "tabgen.musicxml"), opts, "tabs")
            sheet_paths = [os.path.join(tmp_dir, "tabgen.musicxml")]
        pdf_parts = [(lead_notes, "lead"), (rhythm_notes, "rhythm")] if split_lead_rhythm else [(notes, "tabs")]
        for sheet_path, (part_notes, title) in zip(sheet_paths, pdf_parts):
            export_tab_data(part_notes, sheet_path, opts, title, changes, duration)
    except Exception as e:
        log(f"楽譜出力をスキップ: {e}")

    if prepare_playback:
        if backing_path:
            try:
                from .synth import render_tab_mix
                log("ギターオフ伴奏とTABのシンセ音を作成中…", 99)
                playback = render_tab_mix(backing_path, [lead_notes, rhythm_notes],
                                          os.path.join(tmp_dir, 'tabgen_playback.wav'))
                if audio_sources is not None:
                    audio_sources['backing'] = backing_path
                    audio_sources['playback'] = playback
            except (OSError, RuntimeError, ValueError) as error:
                playback_error = f"TABの再生音を作成できませんでした: {error}"
        if playback_error:
            log(playback_error)
            if audio_sources is not None:
                audio_sources['playback_error'] = playback_error
    log("完了!", 100)
    return tab_text, ";".join(midi_paths), ";".join(sheet_paths), stem_path
