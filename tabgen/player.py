"""Audio preview with a playback clock suitable for synchronized TAB display."""
from __future__ import annotations

import math
import time

import soundfile as sf

try:
    import sounddevice as sd
    SD_OK = True
except (ImportError, OSError):
    SD_OK = False


class AudioPlayer:
    def __init__(self):
        self.data = None
        self.sr = 22050
        self.playing = False
        self.on_end = None
        self.last_error = ""
        self._position = 0.
        self._stream = None
        self._session = None
        self._start_frame = 0
        self._offset = 0
        self._dac_start = None
        self._clock_start = 0.

    @property
    def duration(self) -> float:
        return len(self.data) / self.sr if self.data is not None else 0.

    @property
    def position(self) -> float:
        if not self.playing or self._stream is None:
            return self._position
        elapsed = time.perf_counter() - self._clock_start
        if self._dac_start is not None:
            try:
                elapsed = float(self._stream.time) - self._dac_start
            except Exception:
                pass  # A monotonic fallback is used if the host clock is unavailable.
        # Submitted buffers are ahead of audible output; do not use their end
        # as the current position. PortAudio's DAC timestamp includes latency.
        frames = min(self._offset, max(0., elapsed) * self.sr)
        return min(self.duration, (self._start_frame + frames) / self.sr)

    def load(self, path: str) -> bool:
        self.stop()
        self.data = None
        self._position = 0.
        self.last_error = ""
        try:
            self.data, self.sr = sf.read(path, always_2d=True, dtype="float32")
        except (OSError, RuntimeError) as error:
            self.last_error = str(error)
            return False
        return True

    def play(self, from_sec: float | None = None) -> bool:
        target = self.position if from_sec is None else float(from_sec)
        self.stop()
        self.last_error = ""
        if not SD_OK:
            self.last_error = "sounddevice / PortAudio を利用できません"
            return False
        if self.data is None or not len(self.data):
            self.last_error = "再生する音声がありません"
            return False
        if not math.isfinite(target):
            raise ValueError("Playback position must be finite")
        target = max(0., min(target, self.duration))
        if target >= self.duration:
            self._position = self.duration
            return False
        self._start_frame = int(target * self.sr)
        self._position = self._start_frame / self.sr
        self._offset = 0
        self._dac_start = None
        session = object()
        self._session = session
        self.playing = True

        def callback(outdata, frames, time_info, status):
            outdata.fill(0)
            if self._session is not session:
                raise sd.CallbackStop
            if self._dac_start is None:
                try:
                    dac = float(time_info.outputBufferDacTime)
                    current = float(time_info.currentTime)
                    if dac == 0 or not math.isfinite(dac) or not math.isfinite(current):
                        raise ValueError("Host output timestamp is unavailable")
                    self._dac_start = dac
                    delay = max(0., dac - current)
                    self._clock_start = time.perf_counter() + delay
                except (AttributeError, TypeError, ValueError):
                    pass
            first = self._start_frame + self._offset
            n = min(frames, len(self.data) - first)
            if n > 0:
                outdata[:n] = self.data[first:first+n]
                self._offset += n
            if first + n >= len(self.data):
                raise sd.CallbackStop

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=self.data.shape[1], dtype="float32",
                callback=callback, finished_callback=lambda: self._finished(session))
            latency = float(self._stream.latency)
            self._clock_start = time.perf_counter() + latency
            self._stream.start()
            return True
        except Exception as error:
            self.last_error = str(error)
            self.stop()
            return False

    def _finished(self, session):
        if self._session is not session:
            return  # Pause, seek and old streams must not jump to the end.
        self._position = self.duration
        self.playing = False
        self._session = None
        if self.on_end:
            self.on_end()

    def pause(self):
        self.stop()

    def stop(self):
        self._position = self.position
        self.playing = False
        self._session = None
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()  # Discard queued audio on pause/seek, rather than draining it.
            except Exception:
                try:
                    stream.stop()
                except Exception:
                    pass
            try:
                stream.close()
            except Exception:
                pass

    def seek(self, sec: float):
        if not math.isfinite(sec):
            raise ValueError("Playback position must be finite")
        resume = self.playing
        self.stop()
        self._position = max(0., min(sec, self.duration))
        if resume and self._position < self.duration:
            self.play()
