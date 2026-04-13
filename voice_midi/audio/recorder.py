import numpy as np
import threading
import time
from typing import Callable, Optional


class MicRecorder:
    """Records audio from the default microphone using sounddevice."""

    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        self._sr = sample_rate
        self._channels = channels
        self._recording = False
        self._chunks: list = []
        self._stream = None
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self.on_level_update: Optional[Callable[[float, float], None]] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_time(self) -> float:
        if not self._recording:
            return 0.0
        return time.time() - self._start_time

    @property
    def sample_rate(self) -> int:
        return self._sr

    def start(self) -> None:
        if self._recording:
            return

        import sounddevice as sd

        self._chunks = []
        self._recording = True
        self._start_time = time.time()

        def callback(indata, frames, time_info, status):
            if not self._recording:
                return
            with self._lock:
                self._chunks.append(indata[:, 0].copy())

            # Report level for the UI meter
            if self.on_level_update:
                rms = np.sqrt(np.mean(indata ** 2))
                elapsed = time.time() - self._start_time
                self.on_level_update(rms, elapsed)

        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=self._channels,
            callback=callback,
            dtype='float32',
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> Optional[np.ndarray]:
        if not self._recording:
            return None

        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._chunks:
                return None
            audio = np.concatenate(self._chunks)
            self._chunks = []

        return audio

    def cancel(self) -> None:
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._chunks = []
