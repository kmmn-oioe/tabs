"""Audio acquisition: load local files or download from YouTube."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading

import numpy as np
import soundfile as sf


def load_audio_channels(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load/resample audio without discarding channel position information."""
    data, rate = sf.read(path, always_2d=True)
    if rate != sr:
        import scipy.signal
        gcd = np.gcd(rate, sr)
        data = scipy.signal.resample_poly(data, sr // gcd, rate // gcd, axis=0)
    return data.astype(np.float32), sr


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio file, resampling to mono at the given sample rate."""
    data, sr = load_audio_channels(path, sr)
    return data.mean(axis=1), sr


def read_preview_audio(path: str):
    """Read the selected original too, including formats such as AAC/M4A.

    Separation outputs WAV, while the original may need FFmpeg decoding.
    Decode to a temporary file; never fall back to playing a different source.
    """
    try:
        return sf.read(path, always_2d=True, dtype="float32")
    except (OSError, RuntimeError):
        if not os.path.isfile(path):
            raise
    with tempfile.TemporaryDirectory(prefix="tabgen_preview_") as directory:
        wav = os.path.join(directory, "preview.wav")
        cmd = [_ffmpeg_location() or "ffmpeg", "-nostdin", "-v", "error", "-y",
               "-i", os.fspath(path), "-vn", "-c:a", "pcm_f32le", wav]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"元音源のデコードに失敗しました: {error}") from error
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"元音源のデコードに失敗しました: {detail}")
        return sf.read(wav, always_2d=True, dtype="float32")


def _ffmpeg_location() -> str | None:
    """Locate an ffmpeg binary: system PATH first, then imageio-ffmpeg wheel."""
    from shutil import which
    if which("ffmpeg"):
        return None  # yt-dlp finds it on PATH by itself
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def download_youtube(url: str, out_dir: str | None = None,
                     progress=None) -> str:
    """Download the audio track of a YouTube video as wav. Returns file path.

    progress: optional callback(progress_pct: float, message: str).
    """
    out_dir = out_dir or tempfile.mkdtemp(prefix="tabgen_")
    out_base = os.path.join(out_dir, "audio")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "--no-playlist",
        "--progress-template", "download:PROG %(progress._percent_str)s",
        "-o", out_base + ".%(ext)s",
        url,
    ]
    ffloc = _ffmpeg_location()
    if ffloc:
        cmd += ["--ffmpeg-location", ffloc]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    stderr_tail: list[str] = []
    wav = out_base + ".wav"

    def drain_stderr():
        assert proc.stderr
        for line in proc.stderr:
            stderr_tail.append(line.rstrip())
            del stderr_tail[:-50]

    t_err = threading.Thread(target=drain_stderr, daemon=True)
    t_err.start()
    assert proc.stdout
    for line in proc.stdout:
        line = line.strip()
        if progress and line.startswith("PROG:"):
            try:
                pct = float(line[5:].strip().rstrip("%"))
                progress(pct * 0.15, f"YouTubeからダウンロード中… {pct:.0f}%")
            except ValueError:
                pass
    proc.wait()
    t_err.join(timeout=5)
    if progress:
        progress(15, "ダウンロード完了")
    if proc.returncode != 0 or not os.path.exists(wav):
        raise RuntimeError(
            "YouTubeのダウンロードに失敗しました。yt-dlpを更新してください:\n"
            "  pip install -U yt-dlp\n\n" + "\n".join(stderr_tail[-20:])
        )
    return wav
