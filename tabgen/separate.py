"""Optional guitar stem separation using Demucs (requires torch + demucs)."""
from __future__ import annotations

import os


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def separate_guitar(wav_path: str, out_dir: str | None = None) -> str:
    """Run Demucs, return path to the 'other'/'guitar'-ish stem wav.

    Uses htdemucs_6s (6-stem) which has a dedicated guitar stem when available,
    falling back to htdemucs 'other' stem.
    """
    out_dir = out_dir or os.path.join(os.path.dirname(wav_path), "stems")
    import torch, demucs.separate  # noqa: E402

    model = "htdemucs_6s"
    try:
        demucs.separate.main(["-n", model, "-o", out_dir, "--two-stems", "guitar", wav_path])
        name = "guitar"
    except SystemExit:
        raise
    except Exception:
        # fallback: 4-stem, take "other"
        model = "htdemucs"
        demucs.separate.main(["-n", model, "-o", out_dir, wav_path])
        name = "no_vocals" if False else "other"

    base = os.path.splitext(os.path.basename(wav_path))[0]
    stem = os.path.join(out_dir, model, base, f"{name}.wav")
    if not os.path.exists(stem):
        # two-stems mode puts the isolated stem under "no_<name>"
        alt = os.path.join(out_dir, model, base, f"no_{name}.wav")
        if os.path.exists(alt):
            stem = alt
        else:
            raise RuntimeError(f"分離結果が見つかりません: {stem}")
    return stem
