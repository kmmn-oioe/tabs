"""Tempo (BPM) and grid-offset estimation without librosa.

Uses an onset-strength envelope (spectral flux) computed with numpy/scipy,
then:
  - BPM: autocorrelation of the onset envelope over lags corresponding to
    40-240 BPM, with double/half-time preference toward 80-180 BPM.
  - offset: the shift in [0, beat_period) that maximizes envelope energy at
    beat positions, so that quantization grid lines land on actual onsets.
"""
from __future__ import annotations

import numpy as np

from .audio import load_audio


def _onset_envelope(mono: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Spectral-flux onset strength. Returns (envelope, frame_period_sec)."""
    import scipy.signal
    n_fft = 1024
    hop = 256
    win = scipy.signal.get_window("hann", n_fft)
    frames = scipy.signal.stft(mono, nperseg=n_fft, noverlap=n_fft - hop,
                               window=win, padded=False)[2]
    mag = np.abs(frames)
    diff = np.diff(mag, axis=1)
    diff[diff < 0] = 0.0
    env = diff.mean(axis=0)
    env = env / (env.std() + 1e-9)
    # smooth lightly to suppress per-frame jitter
    kernel = np.hanning(5)
    kernel /= kernel.sum()
    env = np.convolve(env, kernel, mode="same")
    frame_period = hop / sr
    return env, frame_period


def estimate_bpm(env: np.ndarray, frame_period: float,
                 min_bpm: float = 60.0, max_bpm: float = 220.0) -> float:
    """Estimate tempo from an onset envelope via autocorrelation."""
    n = len(env)
    if n < 8:
        return 120.0
    env = env - env.mean()
    spec = np.fft.rfft(env, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac /= ac[0] + 1e-9

    min_lag = max(2, int(round(60.0 / max_bpm / frame_period)))
    max_lag = min(n - 1, int(round(60.0 / min_bpm / frame_period)))
    if min_lag >= max_lag:
        return 120.0
    lags = np.arange(min_lag, max_lag + 1)
    vals = ac[min_lag:max_lag + 1]
    best = lags[int(np.argmax(vals))]

    # fold the lag into 60-180 BPM by doubling/halving
    bpm = 60.0 / (best * frame_period)
    while bpm < 80.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0
    return float(bpm)


def estimate_offset(env: np.ndarray, frame_period: float,
                    bpm: float) -> float:
    """Find grid origin in [0, beat_period) maximizing onset energy."""
    beat = 60.0 / bpm
    n = len(env)
    if n < 8:
        return 0.0
    shifts = np.arange(0.0, beat, frame_period)
    best_shift, best_score = 0.0, -1.0
    # half-beat resolution is enough for offset and much more robust
    for i in range(0, len(shifts), 2):
        shift = shifts[i]
        positions = np.arange(shift, n * frame_period, beat)
        idx = np.round(positions / frame_period).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        if len(idx) == 0:
            continue
        score = float(env[idx].mean())
        if score > best_score:
            best_score, best_shift = score, float(shift)
    return best_shift


def estimate_bpm_and_offset(wav_path: str) -> tuple[float, float]:
    """Estimate (bpm, grid_offset_seconds) from an audio file."""
    mono, sr = load_audio(wav_path, sr=22050)
    # cap analysis to the first ~2 minutes for speed
    mono = mono[: 22050 * 120]
    env, frame_period = _onset_envelope(mono, sr)
    if len(env) < 32:
        return 120.0, 0.0
    bpm = estimate_bpm(env, frame_period)
    offset = estimate_offset(env, frame_period, bpm)
    return bpm, offset
