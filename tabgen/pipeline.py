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
) -> tuple[str, str, str, str]:
    """Returns (tab_text, midi_paths, sheet_paths, stem_wav_path).

    source: local audio file path or YouTube URL.
    audio_sources: optional output dictionary containing original/separated paths.
    diagnostics: optional snapshots for offline evaluation; does not affect generation.
    """
    opts = opts or TabOptions()
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
    if use_separation:
        log("ギターパートを分離中…(Demucs・時間がかかります)")
        from .separate import demucs_available, separate_guitar
        if not demucs_available():
            log("※ demucs/torch が未インストールのため分離をスキップします")
        else:
            stem_path = separate_guitar(wav, tmp_dir, progress=progress)
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
            bpm, offset = estimate_bpm_and_offset(stem_path)
            opts.bpm = round(bpm, 1)
            opts.grid_offset = round(offset, 3)
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
        lead_evidence_method = "disabled"
        lead_method = "リード判定: 検出ノートのみ（音声による存在判定は無効）"
        if opts.audio_lead_detection:
            from .lead_audio import analyze_lead_audio
            try:
                log("音声の定位・倍音・発音から独立したリードの根拠を解析中…", 96)
                evidence = analyze_lead_audio(stem_path, notes, min_confidence=opts.min_confidence)
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
        rhythm, changes = arrange_rhythm(notes, opts, changes=changes)
        log(f"リズム伴奏: {len(changes)} 区間のコード切替を作成しました", 97)
        tab_text = (
            "====== リードギター ======\n"
            + lead_method + "\n"
            + entry_text + with_header(lead_text)
            + "\n\n====== リズムギター ======\n"
            + analysis_method + "\n"
            + progression_text(changes, opts) + "\n\n"
            + with_header(notes_to_tab(rhythm, opts, arranged=True))
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
            export_musicxml(rhythm_notes, os.path.join(tmp_dir, "tabgen_rhythm.musicxml"), opts, "rhythm")
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

    log("完了!", 100)
    return tab_text, ";".join(midi_paths), ";".join(sheet_paths), stem_path
