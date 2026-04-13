import numpy as np
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Callable, Optional


class WaveformView(ctk.CTkFrame):
    def __init__(self, parent, on_seek: Optional[Callable[[float], None]] = None):
        super().__init__(parent, height=150)
        self._on_seek = on_seek
        self._duration = 0.0

        self.fig = Figure(figsize=(8, 1.5), dpi=72, facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111)
        self._style_axis()
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.22)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(highlightthickness=0, borderwidth=0)
        widget.pack(fill="both", expand=True)
        self.canvas.draw()

        self._cursor_line = None

        if on_seek:
            self.canvas.mpl_connect('button_press_event', self._on_click)

    def _style_axis(self) -> None:
        self.ax.set_facecolor('#ffffff')
        self.ax.tick_params(colors='#333333', labelsize=7)
        for spine in ('top', 'right'):
            self.ax.spines[spine].set_visible(False)
        for spine in ('bottom', 'left'):
            self.ax.spines[spine].set_color('#999999')
        self.ax.set_ylabel('Amp', color='#333333', fontsize=7)
        self.ax.set_title('Waveform', color='#222222', fontsize=9, pad=3)

    def set_audio(self, samples: np.ndarray, sr: int) -> None:
        self.ax.clear()
        self._style_axis()
        self._duration = len(samples) / sr

        max_points = 4000
        if len(samples) > max_points:
            step = len(samples) // max_points
            display = samples[::step]
        else:
            display = samples
        times = np.linspace(0, self._duration, len(display))

        self.ax.plot(times, display, color='#1565c0', linewidth=0.5, alpha=0.8)
        self.ax.set_xlim(0, self._duration)
        self._cursor_line = self.ax.axvline(x=0, color='#ff5722', linewidth=1, alpha=0.8)
        self.canvas.draw_idle()

    def set_cursor(self, time_seconds: float) -> None:
        if self._cursor_line:
            self._cursor_line.set_xdata([time_seconds])
            self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes == self.ax and self._on_seek and self._duration > 0:
            self._on_seek(max(0, min(event.xdata, self._duration)))

    def clear(self) -> None:
        self.ax.clear()
        self._style_axis()
        self._cursor_line = None
        self.canvas.draw_idle()
