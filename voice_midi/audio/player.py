import numpy as np
import threading
from typing import Callable, Optional


class AudioPlayer:
    def __init__(self):
        self._samples: Optional[np.ndarray] = None
        self._sr: int = 44100
        self._position: int = 0
        self._playing = False
        self._stream = None
        self._lock = threading.Lock()
        self.on_position_changed: Optional[Callable[[float], None]] = None
        self.on_playback_finished: Optional[Callable[[], None]] = None

    def load(self, samples: np.ndarray, sample_rate: int) -> None:
        self.stop()
        self._samples = samples
        self._sr = sample_rate
        self._position = 0

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def current_position(self) -> float:
        if self._sr == 0:
            return 0.0
        return self._position / self._sr

    @property
    def duration(self) -> float:
        if self._samples is None or self._sr == 0:
            return 0.0
        return len(self._samples) / self._sr

    def play(self) -> None:
        if self._samples is None:
            return
        if self._playing:
            return

        import sounddevice as sd

        self._playing = True

        def callback(outdata, frames, time_info, status):
            with self._lock:
                if not self._playing:
                    outdata[:] = 0
                    raise sd.CallbackAbort()

                end = self._position + frames
                if end > len(self._samples):
                    remaining = len(self._samples) - self._position
                    if remaining > 0:
                        outdata[:remaining, 0] = self._samples[self._position:self._position + remaining]
                    outdata[remaining:] = 0
                    self._position = len(self._samples)
                    self._playing = False
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = self._samples[self._position:end]
                    self._position = end

        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            callback=callback,
            finished_callback=self._on_stream_finished,
            dtype='float32',
        )
        self._stream.start()
        self._start_position_updates()

    def pause(self) -> None:
        self._playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def stop(self) -> None:
        self._playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._position = 0

    def seek(self, position_seconds: float) -> None:
        with self._lock:
            self._position = int(position_seconds * self._sr)
            if self._samples is not None:
                self._position = max(0, min(self._position, len(self._samples)))

    def _on_stream_finished(self) -> None:
        self._playing = False
        if self.on_playback_finished:
            self.on_playback_finished()

    def _start_position_updates(self) -> None:
        def updater():
            while self._playing:
                if self.on_position_changed:
                    self.on_position_changed(self.current_position)
                threading.Event().wait(0.05)

        t = threading.Thread(target=updater, daemon=True)
        t.start()
