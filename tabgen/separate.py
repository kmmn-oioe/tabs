"""Optional guitar stem separation using Demucs (requires torch + demucs).

Runs demucs as a subprocess and reports tqdm progress percentages.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _device_args() -> list[str]:
    try:
        import torch
        if torch.cuda.is_available():
            return ["-d", "cuda"]
    except ImportError:
        pass
    return []


def separate_guitar(wav_path: str, out_dir: str | None = None,
                    progress=None, pct_range=(15, 50)) -> str:
    """Run Demucs, return path to the isolated guitar stem wav."""
    out_dir = out_dir or os.path.join(os.path.dirname(wav_path), "stems")
    lo, hi = pct_range

    def report(pct_in_stage: float, msg: str):
        if progress:
            progress(lo + (hi - lo) * pct_in_stage, msg)

    def run(args, label, model):
        cmd = [sys.executable, "-m", "demucs.separate", *args,
               "-o", out_dir, wav_path]
        report(0.0, f"{label}の読み込み中…")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        err_tail = ""
        assert proc.stderr
        buf = b""
        while True:
            chunk = proc.stderr.read(256)
            if not chunk:
                break
            buf += chunk
            *parts, buf = buf.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
            for part in parts:
                line = part.decode("utf-8", errors="replace").rstrip()
                if line:
                    err_tail = line
                    m = re.search(r"(\d+)%\|", line)
                    if m and progress:
                        report(float(m.group(1)) / 100, f"{label}分離中… {m.group(1)}%")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Demucs({model})の実行に失敗しました:\n" + err_tail)

    # prefer 6-stem model with a dedicated guitar stem; fall back to 4-stem
    try:
        run(["-n", "htdemucs_6s", "--two-stems", "guitar", *_device_args()],
            "ギター", "htdemucs_6s")
        model, name = "htdemucs_6s", "no_guitar"
    except RuntimeError:
        report(0.0, "標準モデルで再試行中…")
        run(["-n", "htdemucs", *_device_args()], "ギター", "htdemucs")
        model, name = "htdemucs", "other"

    report(1.0, "分離完了")
    base = os.path.splitext(os.path.basename(wav_path))[0]
    stem = os.path.join(out_dir, model, base, f"{name}.wav")
    if not os.path.exists(stem):
        raise RuntimeError(f"分離結果が見つかりません: {stem}")
    return stem
