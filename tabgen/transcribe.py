"""Note transcription using Spotify's basic-pitch."""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np

from .audio import load_audio


def transcribe(wav_path: str, progress=None, pct_range=(50, 95),
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
