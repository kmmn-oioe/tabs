"""Pipeline: audio/YouTube -> tab text + MIDI."""
from __future__ import annotations

import os
import tempfile
from typing import Callable

from .audio import load_audio, download_youtube
from .tab import TabOptions, notes_to_tab


def generate(
    source: str,
    opts: TabOptions | None = None,
    use_separation: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[str, str, str]:
    """Returns (tab_text, midi_path, stem_wav_path).

    source: local audio file path or YouTube URL.
    """
    opts = opts or TabOptions()

    def log(msg: str):
        if progress:
            progress(msg)

    tmp_dir = tempfile.mkdtemp(prefix="tabgen_")

    # 1. acquire audio
    if source.lower().startswith(("http://", "https://", "www.", "youtube.com", "youtu.be")):
        log("YouTubeから音声をダウンロード中…")
        wav = download_youtube(source, tmp_dir)
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
            stem_path = separate_guitar(wav, tmp_dir)

    # 3. transcribe
    log("音程を解析中…(basic-pitch)")
    from .transcribe import transcribe
    notes = transcribe(stem_path)
    log(f"{len(notes)} ノートを検出しました")

    # 4. build tab
    log("タブ譜を組み立て中…")
    tab_text = notes_to_tab(notes, opts)

    # 5. export MIDI
    midi_path = os.path.join(tmp_dir, "tabgen_output.mid")
    try:
        from .export import export_midi
        export_midi(notes, midi_path)
    except Exception as e:
        log(f"MIDI出力をスキップ: {e}")
        midi_path = ""

    log("完了!")
    return tab_text, midi_path, stem_path
