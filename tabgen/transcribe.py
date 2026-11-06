"""Note transcription using Spotify's basic-pitch."""
from __future__ import annotations

import numpy as np

from .audio import load_audio


def transcribe(wav_path: str):
    """Run basic-pitch. Returns a list of note dicts:
    {start, end, pitch(midi), confidence}
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model_output, midi_data, note_events = predict(
        wav_path,
        ICASSP_2022_MODEL_PATH,
        minimum_note_length=100,   # ms
        melodia_trick=True,
    )
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
