"""Pipeline: audio/YouTube -> tab text + MIDI."""
from __future__ import annotations

import os
import tempfile
from typing import Callable

from .audio import load_audio, download_youtube
from .tab import TabOptions, notes_to_tab, arrange_notes


def generate(
    source: str,
    opts: TabOptions | None = None,
    use_separation: bool = False,
    split_lead_rhythm: bool = False,
    lead_threshold: int = 64,
    min_note_length_ms: float = 70.0,
    auto_bpm: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[str, str, str, str]:
    """Returns (tab_text, midi_paths, sheet_paths, stem_wav_path).

    source: local audio file path or YouTube URL.
    """
    opts = opts or TabOptions()

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
    stem_path = wav
    if use_separation:
        log("ギターパートを分離中…(Demucs・時間がかかります)")
        from .separate import demucs_available, separate_guitar
        if not demucs_available():
            log("※ demucs/torch が未インストールのため分離をスキップします")
        else:
            stem_path = separate_guitar(wav, tmp_dir, progress=progress)

    # 3. transcribe
    log("音程を解析中…", 50)
    from .transcribe import transcribe
    notes = transcribe(stem_path, progress=progress,
                       min_note_length_ms=min_note_length_ms)
    log(f"{len(notes)} ノートを検出しました", 95)

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
    try:
        from .split import split_notes
    except ImportError:
        split_notes = None

    def with_header(text: str) -> str:
        return f"BPM {opts.bpm:g}  [{opts.difficulty}]" + (f"  (カポ {opts.capo})" if opts.capo else "") + "\n" + text

    def arrange(part: list[dict]) -> list[dict]:
        arranged = arrange_notes(part, opts)
        log(f"演奏用に調整 ({opts.difficulty}): {len(part)} → {len(arranged)} ノート", 97)
        return arranged

    if split_lead_rhythm:
        from .rhythm import arrange_rhythm, progression_text
        lead, rhythm = split_notes(notes, lead_threshold)
        log(f"リード {len(lead)} / リズム {len(rhythm)} ノートに振り分けました")
        lead = arrange(lead)
        rhythm, changes = arrange_rhythm(notes, opts)
        log(f"リズム伴奏: {len(changes)} 区間のコード切替を作成しました", 97)
        tab_text = (
            "====== リードギター ======\n"
            + with_header(notes_to_tab(lead, opts, arranged=True))
            + "\n\n====== リズムギター ======\n"
            + progression_text(changes, opts) + "\n\n"
            + with_header(notes_to_tab(rhythm, opts, arranged=True))
        )
        lead_notes, rhythm_notes = lead, rhythm
    else:
        notes = arrange(notes)
        tab_text = with_header(notes_to_tab(notes, opts, arranged=True))
        lead_notes, rhythm_notes = notes, []

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
        log("楽譜(MusicXML)を作成中…", 98)
        if split_lead_rhythm:
            export_musicxml(lead_notes, os.path.join(tmp_dir, "tabgen_lead.musicxml"), opts, "lead")
            export_musicxml(rhythm_notes, os.path.join(tmp_dir, "tabgen_rhythm.musicxml"), opts, "rhythm")
            sheet_paths = [os.path.join(tmp_dir, "tabgen_lead.musicxml"),
                           os.path.join(tmp_dir, "tabgen_rhythm.musicxml")]
        else:
            export_musicxml(notes, os.path.join(tmp_dir, "tabgen.musicxml"), opts, "tabs")
            sheet_paths = [os.path.join(tmp_dir, "tabgen.musicxml")]
    except Exception as e:
        log(f"楽譜出力をスキップ: {e}")

    log("完了!", 100)
    return tab_text, ";".join(midi_paths), ";".join(sheet_paths), stem_path
