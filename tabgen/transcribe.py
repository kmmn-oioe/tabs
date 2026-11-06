"""Note transcription using Spotify's basic-pitch."""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np

from .audio import load_audio, load_audio_channels


def _transcribe_mono(wav_path: str, progress=None, pct_range=(50, 95),
               min_note_length_ms: float = 70.0):
    """Run basic-pitch. Returns a list of note dicts:
    {start, end, pitch(midi), confidence}
    """
    import basic_pitch
    import basic_pitch.inference as bpi
    lo, hi = pct_range

    def report(pct_in_stage: float, msg: str):
        if progress:
            progress(lo + (hi - lo) * pct_in_stage, msg)

    report(0.0, "解析モデルを準備中…")

    # Prefer ONNX Runtime (DirectML GPU on Windows, falls back to CPU)
    # over TensorFlow: much faster and lighter.
    onnx_file = os.path.join(os.path.dirname(basic_pitch.__file__),
                             "saved_models", "icassp_2022", "nmp.onnx")
    model_path = None
    try:
        import onnxruntime as ort
        if os.path.exists(onnx_file):
            # basic-pitch hardcodes providers=["CPUExecutionProvider"];
            # wrap it to prefer the GPU (DirectML) provider.
            class _ORTWithGPU:
                @staticmethod
                def InferenceSession(path, **kw):
                    try:
                        return ort.InferenceSession(
                            path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
                    except Exception:
                        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            import basic_pitch.inference as bpi
            bpi.ort = _ORTWithGPU
            model_path = onnx_file
            logging.getLogger(__name__).info("using ONNX runtime (DirectML GPU) for basic-pitch")
    except ImportError:
        pass
    if model_path is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        model_path = ICASSP_2022_MODEL_PATH

    # Pre-resample to mono 22.05 kHz: ~4x faster than feeding 44.1 kHz stereo,
    # with no effect on note detection.
    mono, sr = load_audio(wav_path, sr=22050)
    tmp = os.path.join(tempfile.gettempdir(), "tabgen_mono.wav")
    try:
        import soundfile as sf
        sf.write(tmp, mono, sr)
        predict_path = tmp
    except Exception:
        predict_path = wav_path  # fall back to the original

    # load the model ourselves so we can wrap predict() and report progress
    model = bpi.Model(model_path)
    sr_loaded, n_samples = 22050, len(mono)
    from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
    hop = AUDIO_N_SAMPLES - 30 * FFT_HOP
    total_windows = max(1, int(np.ceil(n_samples / hop)))
    done_windows = [0]

    orig_predict = model.predict

    def counting_predict(x):
        out = orig_predict(x)
        done_windows[0] += x.shape[0] if hasattr(x, "shape") else 1
        report(min(1.0, done_windows[0] / total_windows),
               f"音程解析中… {min(100, done_windows[0] * 100 // total_windows)}%")
        return out

    model.predict = counting_predict  # type: ignore[method-assign]

    model_output, midi_data, note_events = bpi.predict(
        predict_path,
        model,
        minimum_note_length=min_note_length_ms,   # ms (速いパッセージの取りこぼし防止)
        melodia_trick=True,
    )
    report(1.0, "解析完了")
    notes = []
    for start, end, pitch, conf, _ in note_events:
        notes.append({
            "start": float(start),
            "end": float(end),
            "pitch": int(pitch),
            "confidence": float(conf),
        })
    notes.sort(key=lambda n: n["start"])
    return notes


def merge_transcriptions(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Merge agreeing mid/side detections, retaining repeated note attacks."""
    pitches = {}
    for n in primary + extra:
        pitches.setdefault(n["pitch"], []).append(n)
    result = []
    for notes in pitches.values():
        merged = []
        for n in sorted(notes, key=lambda n: n["start"]):
            if merged:
                previous = merged[-1]
                overlap = min(previous["end"], n["end"]) - max(previous["start"], n["start"])
                shorter = min(previous["end"]-previous["start"], n["end"]-n["start"])
                if abs(n["start"]-previous["start"]) <= .06 and overlap >= .6*max(shorter, .001):
                    if n["confidence"] > previous["confidence"]:
                        merged[-1] = dict(n)
                    continue
            merged.append(dict(n))
        result.extend(merged)
    return sorted(result, key=lambda n: (n["start"], n["pitch"]))


def transcribe(wav_path: str, progress=None, pct_range=(50, 95),
               min_note_length_ms: float = 70., *, stereo_candidates: bool = False):
    """Detect notes from mono, optionally adding significant stereo-difference audio.

    A part with opposite channel phase can disappear from (L+R)/2. The side
    pass only supplies candidates; it is not assumed to contain the lead.
    """
    side = None
    if stereo_candidates:
        channels, sr = load_audio_channels(wav_path)
        if channels.shape[1] == 2 and len(channels):
            mid = channels.mean(axis=1)
            difference = (channels[:, 0]-channels[:, 1])/2
            mid_rms = float(np.sqrt(np.mean(mid**2)))
            side_rms = float(np.sqrt(np.mean(difference**2)))
            if side_rms > max(1e-5, .12*mid_rms):
                side = difference
    if side is None:
        return _transcribe_mono(wav_path, progress, pct_range, min_note_length_ms)
    lo, hi = pct_range
    middle = (lo+hi)/2
    primary = _transcribe_mono(wav_path, progress, (lo, middle), min_note_length_ms)
    if progress:
        progress(middle, "左右差成分の音程候補を追加解析中…")
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="tabgen_side_") as directory:
        path = os.path.join(directory, "side.wav")
        sf.write(path, side, sr, subtype="FLOAT")
        extra = _transcribe_mono(path, progress, (middle, hi), min_note_length_ms)
    return merge_transcriptions(primary, extra)
