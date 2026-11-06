"""Simple audio player for previewing the source while reading the tab."""
from __future__ import annotations

import threading

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
    SD_OK = True
except (ImportError, OSError):
    SD_OK = False


class AudioPlayer:
    """Non-blocking audio player with position callback (poll via .position)."""

    def __init__(self):
        self.data = None
        self.sr = 22050
        self.position = 0.0     # seconds (updated by UI polling)
        self.playing = False
        self.on_end = None
        self._stream = None
        self._frames_played = 0

    @property
    def duration(self) -> float:
        return len(self.data) / self.sr if self.data is not None else 0.0

    def load(self, path: str) -> bool:
        self.stop()
        if not SD_OK:
            return False
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        self.data = data
        self.sr = sr
        self.position = 0.0
        return True

    def play(self, from_sec: float | None = None):
        if self.data is None:
            return
        if self.playing:
            self.stop()
        start = int((from_sec if from_sec is not None else self.position) * self.sr)
        start = max(0, min(start, len(self.data) - 1))
        self._frames_played = start
        chunk = self.data[start:]
        self.playing = True

        def callback(outdata, frames, time_info, status):
            n = min(frames, len(chunk) - self._offset[0])
            if n <= 0:
                outdata[:] = 0
                raise sd.CallbackStop
            outdata[:n] = chunk[self._offset[0]:self._offset[0] + n]
            self._offset[0] += n

        self._offset = [0]
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=self.data.shape[1],
                dtype="float32", callback=callback,
                finished_callback=self._finished)
            self._stream.start()
        except Exception:
            self.playing = False

    def _finished(self):
        self.playing = False
        self.position = self.duration
        if self.on_end:
            self.on_end()

    def pause(self):
        if self.playing and self._stream:
            self._frames_played += self._offset[0]
            self.position = self._frames_played / self.sr
            self.stop()

    def stop(self):
        self.playing = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def seek(self, sec: float):
        self.position = max(0.0, min(sec, self.duration))
        if self.playing:
            self.play(self.position)
