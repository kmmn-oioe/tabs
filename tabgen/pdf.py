"""Convert MusicXML to PDF using MuseScore's command-line interface."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

# standard install locations for MuseScore on Windows
MUSESCORE_CANDIDATES = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
]


def find_musescore() -> str | None:
    from shutil import which
    for name in ("MuseScore4", "MuseScore3", "musescore4", "musescore3"):
        p = which(name)
        if p:
            return p
    for p in MUSESCORE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def musicxml_to_pdf(musicxml_path: str, pdf_path: str | None = None) -> str:
    """Render a .musicxml file to PDF. Returns the pdf path.

    Raises RuntimeError with install instructions if MuseScore is missing.
    """
    exe = find_musescore()
    if not exe:
        raise RuntimeError(
            "PDF出力には MuseScore 4 (無料) が必要です。\n"
            "https://musescore.org/ja からインストールしてください。\n"
            "インストール後、このアプリを再起動すると使えるようになります。")

    out_dir = tempfile.mkdtemp(prefix="tabgen_pdf_")
    tmp_pdf = os.path.join(out_dir, os.path.splitext(os.path.basename(musicxml_path))[0] + ".pdf")
    cmd = [exe, "-o", tmp_pdf, musicxml_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            encoding="utf-8", errors="replace")
    if not os.path.exists(tmp_pdf):
        raise RuntimeError("PDF変換に失敗しました:\n" + (result.stdout + result.stderr)[-1500:])

    if pdf_path is None:
        pdf_path = tmp_pdf
    else:
        with open(tmp_pdf, "rb") as src, open(pdf_path, "wb") as dst:
            dst.write(src.read())
    return pdf_path
