import customtkinter as ctk
from typing import Callable, Optional


class TransportBar(ctk.CTkFrame):
    def __init__(
        self,
        parent,
        on_open: Callable,
        on_record_toggle: Callable,
        on_play_original: Callable,
        on_play_midi: Callable,
        on_stop: Callable,
    ):
        super().__init__(parent)
        self._on_open = on_open
        self._on_record_toggle = on_record_toggle
        self._on_play_original = on_play_original
        self._on_play_midi = on_play_midi
        self._on_stop = on_stop
        self._duration = 0.0
        self._is_recording = False

        self.grid_columnconfigure(4, weight=1)

        # --- Row: file + record + playback + progress ---
        # Open file
        self.open_btn = ctk.CTkButton(
            self, text="Open File", width=90, command=self._on_open,
        )
        self.open_btn.grid(row=0, column=0, padx=(10, 4), pady=8)

        # Record button
        self.record_btn = ctk.CTkButton(
            self, text="Record", width=90,
            fg_color="#c62828", hover_color="#e53935",
            command=self._on_record_toggle,
        )
        self.record_btn.grid(row=0, column=1, padx=4, pady=8)

        # File / recording label
        self.file_label = ctk.CTkLabel(self, text="No file loaded", anchor="w", width=160)
        self.file_label.grid(row=0, column=2, padx=5, pady=8, sticky="w")

        # Playback controls
        ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctrl_frame.grid(row=0, column=3, padx=6, pady=8)

        self.play_orig_btn = ctk.CTkButton(
            ctrl_frame, text="Play Original", width=105,
            command=self._on_play_original, state="disabled",
        )
        self.play_orig_btn.grid(row=0, column=0, padx=2)

        self.play_midi_btn = ctk.CTkButton(
            ctrl_frame, text="Play MIDI", width=80,
            command=self._on_play_midi, state="disabled",
        )
        self.play_midi_btn.grid(row=0, column=1, padx=2)

        self.stop_btn = ctk.CTkButton(
            ctrl_frame, text="Stop", width=55,
            command=self._on_stop, state="disabled",
        )
        self.stop_btn.grid(row=0, column=2, padx=2)

        # Progress bar
        self.progress = ctk.CTkProgressBar(self, width=180)
        self.progress.grid(row=0, column=4, padx=8, pady=8, sticky="ew")
        self.progress.set(0)

        # Time display
        self.time_label = ctk.CTkLabel(self, text="0:00 / 0:00", width=95)
        self.time_label.grid(row=0, column=5, padx=(4, 10), pady=8)

    # --- Public API ---

    def set_file_name(self, name: str, duration: float) -> None:
        self.file_label.configure(text=name)
        self._duration = duration
        self.play_orig_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        self.update_position(0.0)

    def enable_midi_playback(self) -> None:
        self.play_midi_btn.configure(state="normal")

    def update_position(self, seconds: float) -> None:
        if self._duration > 0:
            self.progress.set(min(seconds / self._duration, 1.0))
        cur = _format_time(seconds)
        total = _format_time(self._duration)
        self.time_label.configure(text=f"{cur} / {total}")

    def set_recording(self, is_recording: bool) -> None:
        self._is_recording = is_recording
        if is_recording:
            self.record_btn.configure(
                text="Stop Rec",
                fg_color="#d50000", hover_color="#ff1744",
            )
            self.open_btn.configure(state="disabled")
            self.file_label.configure(text="Recording...")
        else:
            self.record_btn.configure(
                text="Record",
                fg_color="#c62828", hover_color="#e53935",
            )
            self.open_btn.configure(state="normal")

    def update_recording_time(self, elapsed: float, level: float) -> None:
        self.time_label.configure(text=f"REC {_format_time(elapsed)}")
        self.progress.set(min(level * 5.0, 1.0))  # level meter


def _format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"
