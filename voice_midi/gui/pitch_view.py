import numpy as np
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from typing import List, Optional

from ..analysis.pitch_detector import PitchResult
from ..analysis.music_utils import NoteEvent, midi_to_note_name, frequency_to_midi


NOTE_COLORS = [
    '#e53935', '#ff7043', '#ffa726', '#ffca28', '#66bb6a', '#26a69a',
    '#29b6f6', '#42a5f5', '#5c6bc0', '#7e57c2', '#ab47bc', '#ec407a',
]


class PitchView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, height=200)

        self.fig = Figure(figsize=(8, 2.2), dpi=72, facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111)
        self._style_axis()
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(highlightthickness=0, borderwidth=0)
        widget.pack(fill="both", expand=True)
        self.canvas.draw()

        self._cursor_line = None

    def _style_axis(self) -> None:
        self.ax.set_facecolor('#0f0f1a')
        self.ax.tick_params(colors='#888888', labelsize=7)
        for spine in ('top', 'right'):
            self.ax.spines[spine].set_visible(False)
        for spine in ('bottom', 'left'):
            self.ax.spines[spine].set_color('#333344')
        self.ax.set_ylabel('Note', color='#888888', fontsize=7)
        self.ax.set_title('Pitch Analysis', color='#cccccc', fontsize=9, pad=3)

    def set_data(
        self,
        notes: List[NoteEvent],
        pitch_result: Optional[PitchResult],
        duration: float,
    ) -> None:
        self.ax.clear()
        self._style_axis()

        if not notes:
            self.ax.set_title('No notes detected', color='#cccccc', fontsize=9, pad=3)
            self.canvas.draw_idle()
            return

        midi_notes = [n.midi_note for n in notes]
        min_note = min(midi_notes) - 3
        max_note = max(midi_notes) + 3

        # Grid: subtle horizontal lines for each semitone
        for midi_val in range(int(min_note), int(max_note) + 1):
            is_black = midi_val % 12 in (1, 3, 6, 8, 10)
            if is_black:
                self.ax.axhspan(
                    midi_val - 0.5, midi_val + 0.5,
                    alpha=0.08, color='#ffffff', zorder=0,
                )
            if midi_val % 12 == 0:
                self.ax.axhline(
                    y=midi_val, color='#333355', linewidth=0.5, alpha=0.6, zorder=0,
                )

        # Note labels on Y axis (C notes only)
        yticks = [n for n in range(int(min_note), int(max_note) + 1) if n % 12 == 0]
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels([midi_to_note_name(n) for n in yticks])

        # Semi-transparent note rectangles as background
        for note in notes:
            color = NOTE_COLORS[note.midi_note % 12]
            alpha = 0.12 + 0.18 * note.confidence
            rect = Rectangle(
                (note.start_time, note.midi_note - 0.4),
                note.duration, 0.8,
                facecolor=color, edgecolor=color,
                linewidth=0.3, alpha=alpha, zorder=1,
            )
            self.ax.add_patch(rect)

        # Draw actual pitch curves per note — this is the key visual
        for note in notes:
            if not note.pitch_curve:
                continue
            curve_t = []
            curve_m = []
            for t, f in note.pitch_curve:
                if f > 0:
                    curve_t.append(t)
                    curve_m.append(frequency_to_midi(f))
            if len(curve_t) >= 2:
                color = NOTE_COLORS[note.midi_note % 12]
                self.ax.plot(
                    curve_t, curve_m, color=color,
                    linewidth=2.0, alpha=0.9, solid_capstyle='round', zorder=3,
                )
            elif len(curve_t) == 1:
                color = NOTE_COLORS[note.midi_note % 12]
                self.ax.plot(
                    curve_t, curve_m, 'o', color=color,
                    markersize=3, alpha=0.9, zorder=3,
                )

        # Also draw raw pitch data lightly if available
        if pitch_result is not None:
            raw_t = pitch_result.times
            raw_f = pitch_result.frequencies
            raw_v = pitch_result.voiced_flags
            raw_p = pitch_result.voiced_probs

            # Only draw voiced frames
            mask = raw_v & (raw_f > 0) & (raw_p > 0.3)
            if np.any(mask):
                raw_midi = np.zeros_like(raw_f)
                raw_midi[mask] = 69.0 + 12.0 * np.log2(raw_f[mask] / 440.0)
                # Draw as faint dots
                self.ax.scatter(
                    raw_t[mask], raw_midi[mask],
                    s=0.3, color='#ffffff', alpha=0.15, zorder=2,
                )

        self.ax.set_xlim(0, duration)
        self.ax.set_ylim(min_note, max_note)
        self.ax.set_title(
            f'Pitch Analysis ({len(notes)} notes)',
            color='#cccccc', fontsize=9, pad=3,
        )

        # Playback cursor
        self._cursor_line = self.ax.axvline(
            x=0, color='#ff5722', linewidth=1.2, alpha=0.9, zorder=10,
        )
        self.canvas.draw_idle()

    def set_notes(self, notes: List[NoteEvent], duration: float) -> None:
        """Backward-compatible wrapper."""
        self.set_data(notes, None, duration)

    def set_cursor(self, time_seconds: float) -> None:
        if self._cursor_line is not None:
            self._cursor_line.set_xdata([time_seconds])
            self.canvas.draw_idle()

    def clear(self) -> None:
        self.ax.clear()
        self._style_axis()
        self._cursor_line = None
        self.canvas.draw_idle()
