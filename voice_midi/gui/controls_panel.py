import customtkinter as ctk
from typing import Callable, Dict, Any

from ..analysis.music_utils import (
    NOTE_NAMES, SCALE_INTERVALS, GM_INSTRUMENTS,
)


class ControlsPanel(ctk.CTkScrollableFrame):
    def __init__(
        self,
        parent,
        on_detect: Callable,
        on_generate: Callable,
        on_export: Callable,
    ):
        super().__init__(parent, width=260)
        self._on_detect = on_detect
        self._on_generate = on_generate
        self._on_export = on_export

        # === Detection Settings ===
        self._section("Detection Settings")

        self._add_label("Min Note Duration (s)")
        dur_frame = ctk.CTkFrame(self, fg_color="transparent")
        dur_frame.pack(fill="x", padx=15, pady=(0, 5))
        self.min_duration_var = ctk.DoubleVar(value=0.08)
        self.min_duration_slider = ctk.CTkSlider(
            dur_frame, from_=0.01, to=0.5, variable=self.min_duration_var,
            number_of_steps=49,
        )
        self.min_duration_slider.pack(side="left", fill="x", expand=True)
        self.min_duration_readout = ctk.CTkLabel(dur_frame, text="0.08", width=40)
        self.min_duration_readout.pack(side="right", padx=(5, 0))
        self.min_duration_var.trace_add("write", lambda *_: self.min_duration_readout.configure(
            text=f"{self.min_duration_var.get():.2f}"))

        self._add_label("Voiced Threshold")
        vt_frame = ctk.CTkFrame(self, fg_color="transparent")
        vt_frame.pack(fill="x", padx=15, pady=(0, 5))
        self.voiced_thresh_var = ctk.DoubleVar(value=0.5)
        self.voiced_thresh_slider = ctk.CTkSlider(
            vt_frame, from_=0.0, to=1.0, variable=self.voiced_thresh_var,
            number_of_steps=20,
        )
        self.voiced_thresh_slider.pack(side="left", fill="x", expand=True)
        self.voiced_thresh_readout = ctk.CTkLabel(vt_frame, text="0.50", width=40)
        self.voiced_thresh_readout.pack(side="right", padx=(5, 0))
        self.voiced_thresh_var.trace_add("write", lambda *_: self.voiced_thresh_readout.configure(
            text=f"{self.voiced_thresh_var.get():.2f}"))

        self._add_label("Pitch Range")
        range_frame = ctk.CTkFrame(self, fg_color="transparent")
        range_frame.pack(fill="x", padx=15, pady=(0, 10))
        ctk.CTkLabel(range_frame, text="Low:").pack(side="left")
        self.fmin_var = ctk.StringVar(value="C2")
        ctk.CTkOptionMenu(
            range_frame, variable=self.fmin_var,
            values=[f"{n}{o}" for o in range(1, 7) for n in NOTE_NAMES],
            width=70,
        ).pack(side="left", padx=5)
        ctk.CTkLabel(range_frame, text="High:").pack(side="left", padx=(10, 0))
        self.fmax_var = ctk.StringVar(value="C7")
        ctk.CTkOptionMenu(
            range_frame, variable=self.fmax_var,
            values=[f"{n}{o}" for o in range(1, 8) for n in NOTE_NAMES],
            width=70,
        ).pack(side="left", padx=5)

        # === Quantization ===
        self._section("Quantization")

        self._add_label("Scale")
        self.scale_var = ctk.StringVar(value="chromatic")
        ctk.CTkOptionMenu(
            self, variable=self.scale_var,
            values=list(SCALE_INTERVALS.keys()),
        ).pack(fill="x", padx=15, pady=(0, 5))

        self._add_label("Root Note")
        self.root_var = ctk.StringVar(value="C")
        ctk.CTkOptionMenu(
            self, variable=self.root_var,
            values=NOTE_NAMES,
        ).pack(fill="x", padx=15, pady=(0, 10))

        # === MIDI Output ===
        self._section("MIDI Output")

        self._add_label("Tempo (BPM)")
        self.tempo_var = ctk.IntVar(value=120)
        ctk.CTkEntry(self, textvariable=self.tempo_var, width=80).pack(
            anchor="w", padx=15, pady=(0, 5))

        self._add_label("Instrument")
        instrument_names = list(GM_INSTRUMENTS.keys())[:32]
        self.instrument_var = ctk.StringVar(value="Acoustic Grand Piano")
        ctk.CTkOptionMenu(
            self, variable=self.instrument_var,
            values=instrument_names,
        ).pack(fill="x", padx=15, pady=(0, 5))

        self._add_label("Velocity")
        self.auto_vel_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self, text="Auto (from amplitude)",
            variable=self.auto_vel_var,
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # === Actions ===
        ctk.CTkFrame(self, height=2, fg_color="#444444").pack(
            fill="x", padx=10, pady=(5, 10))

        self.detect_btn = ctk.CTkButton(
            self, text="Detect Pitch", command=self._on_detect,
            fg_color="#1565c0", hover_color="#1976d2",
            height=36, state="disabled",
        )
        self.detect_btn.pack(fill="x", padx=15, pady=4)

        self.generate_btn = ctk.CTkButton(
            self, text="Generate MIDI", command=self._on_generate,
            fg_color="#2e7d32", hover_color="#388e3c",
            height=36, state="disabled",
        )
        self.generate_btn.pack(fill="x", padx=15, pady=4)

        self.export_btn = ctk.CTkButton(
            self, text="Export MIDI", command=self._on_export,
            fg_color="#e65100", hover_color="#ef6c00",
            height=36, state="disabled",
        )
        self.export_btn.pack(fill="x", padx=15, pady=(4, 15))

    # ── Layout helpers ──

    def _section(self, title: str) -> None:
        ctk.CTkLabel(self, text=title, font=("", 14, "bold")).pack(
            anchor="w", padx=10, pady=(15, 5))

    def _add_label(self, text: str) -> None:
        ctk.CTkLabel(self, text=text, font=("", 11)).pack(
            anchor="w", padx=15, pady=(5, 2))

    # ── Public API ──

    def enable_detection(self) -> None:
        self.detect_btn.configure(state="normal")

    def enable_generation(self) -> None:
        self.generate_btn.configure(state="normal")

    def enable_export(self) -> None:
        self.export_btn.configure(state="normal")

    def get_settings(self) -> Dict[str, Any]:
        from ..analysis.music_utils import note_name_to_midi, midi_to_frequency

        fmin_midi = note_name_to_midi(self.fmin_var.get())
        fmax_midi = note_name_to_midi(self.fmax_var.get())

        scale = self.scale_var.get()
        scale_filter = None
        if scale != "chromatic":
            scale_filter = (self.root_var.get(), scale)

        velocity = None
        if not self.auto_vel_var.get():
            velocity = 100

        return {
            'min_note_duration': self.min_duration_var.get(),
            'voiced_threshold': self.voiced_thresh_var.get(),
            'fmin': midi_to_frequency(fmin_midi),
            'fmax': midi_to_frequency(fmax_midi),
            'tempo': max(20, min(300, self.tempo_var.get())),
            'instrument': GM_INSTRUMENTS.get(self.instrument_var.get(), 0),
            'default_velocity': velocity,
            'scale_filter': scale_filter,
        }
