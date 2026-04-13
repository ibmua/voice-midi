import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

from ..pipeline import ConversionPipeline
from ..audio.player import AudioPlayer
from ..audio.recorder import MicRecorder


class VoiceMidiApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("Voice to MIDI")
        self.geometry("1100x750")
        self.minsize(800, 500)

        self.pipeline = ConversionPipeline()
        self.player = AudioPlayer()
        self.player.on_position_changed = self._on_position_update
        self.player.on_playback_finished = self._on_playback_done

        self.recorder = MicRecorder(sample_rate=22050)
        self.recorder.on_level_update = self._on_rec_level

        self._current_file = None
        self._playing_midi = False

        self._build_ui()

    def _build_ui(self) -> None:
        # ── Top toolbar ──
        toolbar = ctk.CTkFrame(self)
        toolbar.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkButton(toolbar, text="Open File", width=90,
                       command=self._open_file).pack(side="left", padx=4, pady=6)
        self.record_btn = ctk.CTkButton(toolbar, text="Record", width=90,
                                         fg_color="#c62828", hover_color="#e53935",
                                         command=self._toggle_recording)
        self.record_btn.pack(side="left", padx=4, pady=6)

        self.file_label = ctk.CTkLabel(toolbar, text="No file loaded", anchor="w")
        self.file_label.pack(side="left", padx=8, pady=6)

        # Time display (right side)
        self.time_label = ctk.CTkLabel(toolbar, text="0:00 / 0:00", width=90)
        self.time_label.pack(side="right", padx=8, pady=6)

        self.progress = ctk.CTkProgressBar(toolbar, width=160)
        self.progress.pack(side="right", padx=4, pady=6)
        self.progress.set(0)

        self.stop_btn = ctk.CTkButton(toolbar, text="Stop", width=55,
                                       command=self._stop, state="disabled")
        self.stop_btn.pack(side="right", padx=2, pady=6)
        self.play_midi_btn = ctk.CTkButton(toolbar, text="Play MIDI", width=80,
                                            command=self._play_midi, state="disabled")
        self.play_midi_btn.pack(side="right", padx=2, pady=6)
        self.play_orig_btn = ctk.CTkButton(toolbar, text="Play Original", width=100,
                                            command=self._play_original, state="disabled")
        self.play_orig_btn.pack(side="right", padx=2, pady=6)

        # ── Main content: left (plots) + right (controls) ──
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        # Left: visualization
        viz = ctk.CTkFrame(body, fg_color="transparent")
        viz.pack(side="left", fill="both", expand=True)

        from .waveform_view import WaveformView
        self.waveform = WaveformView(viz, on_seek=self._seek)
        self.waveform.pack(fill="both", expand=True, pady=(0, 2))

        from .pitch_view import PitchView
        self.pitch_view = PitchView(viz)
        self.pitch_view.pack(fill="both", expand=True, pady=(2, 0))

        # Right: controls sidebar
        sidebar = ctk.CTkScrollableFrame(body, width=240)
        sidebar.pack(side="right", fill="y", padx=(6, 0))
        self._build_controls(sidebar)

        # ── Status bar ──
        status_bar = ctk.CTkFrame(self, height=28)
        status_bar.pack(fill="x", padx=8, pady=(4, 8))

        self.status_label = ctk.CTkLabel(status_bar, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=10)
        self.notes_label = ctk.CTkLabel(status_bar, text="", anchor="e")
        self.notes_label.pack(side="right", padx=10)

    def _build_controls(self, parent) -> None:
        from ..analysis.music_utils import NOTE_NAMES, SCALE_INTERVALS, GM_INSTRUMENTS

        def section(text):
            ctk.CTkLabel(parent, text=text, font=("", 13, "bold")).pack(
                anchor="w", padx=8, pady=(12, 4))

        def label(text):
            ctk.CTkLabel(parent, text=text, font=("", 11)).pack(
                anchor="w", padx=12, pady=(4, 1))

        # Detection
        section("Detection")

        label("Min Note Duration (s)")
        f1 = ctk.CTkFrame(parent, fg_color="transparent")
        f1.pack(fill="x", padx=12, pady=(0, 4))
        self.min_dur_var = ctk.DoubleVar(value=0.08)
        ctk.CTkSlider(f1, from_=0.01, to=0.5, variable=self.min_dur_var,
                       number_of_steps=49).pack(side="left", fill="x", expand=True)
        self._dur_lbl = ctk.CTkLabel(f1, text="0.08", width=36)
        self._dur_lbl.pack(side="right", padx=(4, 0))
        self.min_dur_var.trace_add("write", lambda *_: self._dur_lbl.configure(
            text=f"{self.min_dur_var.get():.2f}"))

        label("Voiced Threshold")
        f2 = ctk.CTkFrame(parent, fg_color="transparent")
        f2.pack(fill="x", padx=12, pady=(0, 4))
        self.voiced_var = ctk.DoubleVar(value=0.5)
        ctk.CTkSlider(f2, from_=0.0, to=1.0, variable=self.voiced_var,
                       number_of_steps=20).pack(side="left", fill="x", expand=True)
        self._vt_lbl = ctk.CTkLabel(f2, text="0.50", width=36)
        self._vt_lbl.pack(side="right", padx=(4, 0))
        self.voiced_var.trace_add("write", lambda *_: self._vt_lbl.configure(
            text=f"{self.voiced_var.get():.2f}"))

        label("Pitch Range")
        f3 = ctk.CTkFrame(parent, fg_color="transparent")
        f3.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(f3, text="Low").pack(side="left")
        self.fmin_var = ctk.StringVar(value="C2")
        ctk.CTkOptionMenu(f3, variable=self.fmin_var,
                           values=[f"{n}{o}" for o in range(1, 7) for n in NOTE_NAMES],
                           width=65).pack(side="left", padx=4)
        ctk.CTkLabel(f3, text="Hi").pack(side="left", padx=(6, 0))
        self.fmax_var = ctk.StringVar(value="C7")
        ctk.CTkOptionMenu(f3, variable=self.fmax_var,
                           values=[f"{n}{o}" for o in range(1, 8) for n in NOTE_NAMES],
                           width=65).pack(side="left", padx=4)

        # Quantize
        section("Quantize")

        label("Scale")
        self.scale_var = ctk.StringVar(value="chromatic")
        ctk.CTkOptionMenu(parent, variable=self.scale_var,
                           values=list(SCALE_INTERVALS.keys())).pack(
            fill="x", padx=12, pady=(0, 4))

        label("Root Note")
        self.root_var = ctk.StringVar(value="C")
        ctk.CTkOptionMenu(parent, variable=self.root_var,
                           values=NOTE_NAMES).pack(fill="x", padx=12, pady=(0, 8))

        # MIDI
        section("MIDI Output")

        label("Tempo (BPM)")
        self.tempo_var = ctk.IntVar(value=120)
        ctk.CTkEntry(parent, textvariable=self.tempo_var, width=70).pack(
            anchor="w", padx=12, pady=(0, 4))

        label("Instrument")
        inst_names = list(GM_INSTRUMENTS.keys())[:32]
        self.inst_var = ctk.StringVar(value="Acoustic Grand Piano")
        ctk.CTkOptionMenu(parent, variable=self.inst_var,
                           values=inst_names).pack(fill="x", padx=12, pady=(0, 4))

        self.auto_vel = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(parent, text="Auto velocity",
                         variable=self.auto_vel).pack(anchor="w", padx=12, pady=(2, 10))

        # Buttons
        ctk.CTkFrame(parent, height=1, fg_color="#555").pack(fill="x", padx=8, pady=4)

        self.detect_btn = ctk.CTkButton(parent, text="Detect Pitch",
                                         fg_color="#1565c0", hover_color="#1976d2",
                                         height=34, state="disabled",
                                         command=self._detect_pitch)
        self.detect_btn.pack(fill="x", padx=12, pady=3)

        self.gen_btn = ctk.CTkButton(parent, text="Generate MIDI",
                                      fg_color="#2e7d32", hover_color="#388e3c",
                                      height=34, state="disabled",
                                      command=self._generate_midi)
        self.gen_btn.pack(fill="x", padx=12, pady=3)

        self.export_btn = ctk.CTkButton(parent, text="Export MIDI",
                                         fg_color="#e65100", hover_color="#ef6c00",
                                         height=34, state="disabled",
                                         command=self._export_midi)
        self.export_btn.pack(fill="x", padx=12, pady=(3, 12))

        self._gm = GM_INSTRUMENTS

    # ── Helpers ──

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _fmt(self, s: float) -> str:
        return f"{int(s)//60}:{int(s)%60:02d}"

    def _get_settings(self):
        from ..analysis.music_utils import note_name_to_midi, midi_to_frequency
        scale = self.scale_var.get()
        sf = (self.root_var.get(), scale) if scale != "chromatic" else None
        return {
            'min_note_duration': self.min_dur_var.get(),
            'voiced_threshold': self.voiced_var.get(),
            'fmin': midi_to_frequency(note_name_to_midi(self.fmin_var.get())),
            'fmax': midi_to_frequency(note_name_to_midi(self.fmax_var.get())),
            'tempo': max(20, min(300, self.tempo_var.get())),
            'instrument': self._gm.get(self.inst_var.get(), 0),
            'default_velocity': None if self.auto_vel.get() else 100,
            'scale_filter': sf,
        }

    # ── Recording ──

    def _toggle_recording(self) -> None:
        if self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self.player.stop()
        self.record_btn.configure(text="Stop Rec", fg_color="#d50000")
        self.file_label.configure(text="Recording...")
        self._set_status("Recording from microphone...")
        try:
            self.recorder.start()
        except Exception as e:
            self.record_btn.configure(text="Record", fg_color="#c62828")
            messagebox.showerror("Error", f"Mic error: {e}")

    def _stop_recording(self) -> None:
        audio = self.recorder.stop()
        self.record_btn.configure(text="Record", fg_color="#c62828")

        if audio is None or len(audio) < 1000:
            self._set_status("Recording too short")
            self.file_label.configure(text="No file loaded")
            return

        sr = self.recorder.sample_rate
        self.pipeline.audio_data = audio
        self.pipeline.sample_rate = sr
        self.pipeline.pitch_result = None
        self.pipeline.notes = None
        self.pipeline.midi_file = None
        self._current_file = None

        dur = len(audio) / sr
        self._duration = dur
        self.file_label.configure(text=f"Recording ({dur:.1f}s)")
        self.waveform.set_audio(audio, sr)
        self.pitch_view.clear()
        self.detect_btn.configure(state="normal")
        self.play_orig_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        self.player.load(audio, sr)
        self._set_status(f"Recorded {dur:.1f}s of audio")
        self.notes_label.configure(text="")

    def _on_rec_level(self, level: float, elapsed: float) -> None:
        self.after(0, lambda: self.time_label.configure(text=f"REC {self._fmt(elapsed)}"))

    # ── File loading ──

    def _open_file(self) -> None:
        from ..audio.loader import get_format_filter
        path = filedialog.askopenfilename(title="Open Audio File",
                                          filetypes=get_format_filter())
        if not path:
            return
        self._current_file = path
        self._set_status(f"Loading {os.path.basename(path)}...")
        self.update()

        def work():
            try:
                self.pipeline.load(path)
                self.after(0, lambda: self._on_loaded(path))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_loaded(self, path: str) -> None:
        name = os.path.basename(path)
        dur = len(self.pipeline.audio_data) / self.pipeline.sample_rate
        self._duration = dur
        self.file_label.configure(text=f"{name} ({dur:.1f}s)")
        self.waveform.set_audio(self.pipeline.audio_data, self.pipeline.sample_rate)
        self.pitch_view.clear()
        self.detect_btn.configure(state="normal")
        self.play_orig_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        self.notes_label.configure(text="")
        self.player.load(self.pipeline.audio_data, self.pipeline.sample_rate)
        self._set_status(f"Loaded: {name}")

    # ── Pitch detection ──

    def _detect_pitch(self) -> None:
        if self.pipeline.audio_data is None:
            return
        s = self._get_settings()
        self._set_status("Detecting pitch...")
        self.detect_btn.configure(state="disabled")
        self.update()

        def work():
            try:
                self.pipeline.run_pitch_detection(fmin=s['fmin'], fmax=s['fmax'])
                self.pipeline.run_note_segmentation(
                    min_note_duration=s['min_note_duration'],
                    voiced_threshold=s['voiced_threshold'])
                self.after(0, self._on_detected)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self.detect_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def _on_detected(self) -> None:
        dur = len(self.pipeline.audio_data) / self.pipeline.sample_rate
        n = len(self.pipeline.notes)
        self.pitch_view.set_data(self.pipeline.notes, self.pipeline.pitch_result, dur)
        self.detect_btn.configure(state="normal")
        self.gen_btn.configure(state="normal")
        self.notes_label.configure(text=f"Notes: {n}")
        self._set_status(f"Detected {n} notes")

    # ── MIDI ──

    def _generate_midi(self) -> None:
        if not self.pipeline.notes:
            return
        s = self._get_settings()
        self._set_status("Generating MIDI...")
        try:
            self.pipeline.run_midi_generation(
                tempo=s['tempo'], instrument=s['instrument'],
                default_velocity=s['default_velocity'], scale_filter=s['scale_filter'])
            self.export_btn.configure(state="normal")
            self.play_midi_btn.configure(state="normal")
            self._set_status("MIDI ready — export or preview")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_midi(self) -> None:
        if not self.pipeline.midi_file:
            return
        default = "recording.mid"
        if self._current_file:
            default = os.path.splitext(os.path.basename(self._current_file))[0] + ".mid"
        path = filedialog.asksaveasfilename(
            title="Export MIDI", defaultextension=".mid", initialfile=default,
            filetypes=[("MIDI", "*.mid"), ("All", "*.*")])
        if path:
            self.pipeline.export_midi(path)
            self._set_status(f"Exported: {os.path.basename(path)}")

    # ── Playback ──

    def _play_original(self) -> None:
        if self.pipeline.audio_data is None:
            return
        self._playing_midi = False
        self.player.load(self.pipeline.audio_data, self.pipeline.sample_rate)
        self.player.play()
        self._set_status("Playing original...")

    def _play_midi(self) -> None:
        if not self.pipeline.notes:
            return
        self._set_status("Rendering preview...")
        self.update()

        def work():
            try:
                preview = self.pipeline.render_preview()
                self.after(0, lambda: self._start_midi(preview))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _start_midi(self, audio) -> None:
        self._playing_midi = True
        self.player.load(audio, 44100)
        self.player.play()
        self._set_status("Playing MIDI preview...")

    def _stop(self) -> None:
        self.player.stop()
        self._set_status("Stopped")

    def _seek(self, pos: float) -> None:
        self.player.seek(pos)
        self.waveform.set_cursor(pos)

    def _on_position_update(self, pos: float) -> None:
        self.after(0, lambda p=pos: self._update_pos(p))

    def _update_pos(self, pos: float) -> None:
        dur = getattr(self, '_duration', 0)
        if dur > 0:
            self.progress.set(min(pos / dur, 1.0))
        self.time_label.configure(text=f"{self._fmt(pos)} / {self._fmt(dur)}")
        self.pitch_view.set_cursor(pos)
        if not self._playing_midi:
            self.waveform.set_cursor(pos)

    def _on_playback_done(self) -> None:
        self.after(0, lambda: self._set_status("Playback finished"))
