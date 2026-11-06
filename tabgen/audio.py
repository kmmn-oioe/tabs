"""Audio acquisition: load local files or download from YouTube."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio file, resampling to mono at the given sample rate."""
    data, rate = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    if rate != sr:
        import scipy.signal
        gcd = np.gcd(rate, sr)
        mono = scipy.signal.resample_poly(mono, sr // gcd, rate // gcd)
    return mono.astype(np.float32), sr


def download_youtube(url: str, out_dir: str | None = None) -> str:
    """Download the audio track of a YouTube video as wav. Returns file path."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="tabgen_")
    out_base = os.path.join(out_dir, "audio")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "--no-playlist",
        "-o", out_base + ".%(ext)s",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wav = out_base + ".wav"
    if result.returncode != 0 or not os.path.exists(wav):
        raise RuntimeError(
            "YouTubeのダウンロードに失敗しました。yt-dlpを更新してください:\n"
            f"  pip install -U yt-dlp\n\n{result.stderr[-2000:]}"
        )
    return wav
