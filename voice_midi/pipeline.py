from typing import Callable, List, Optional, Tuple

import numpy as np
from midiutil import MIDIFile

from .audio.loader import load_audio
from .analysis.pitch_detector import PitchResult, detect_pitch
from .analysis.note_segmenter import segment_notes
from .analysis.music_utils import NoteEvent
from .midi.generator import generate_midi, save_midi
from .midi.synthesizer import render_notes_to_audio


class ConversionPipeline:
    def __init__(self):
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.pitch_result: Optional[PitchResult] = None
        self.notes: Optional[List[NoteEvent]] = None
        self.midi_file: Optional[MIDIFile] = None
        self.on_progress: Optional[Callable[[str, float], None]] = None

    def _report(self, stage: str, progress: float) -> None:
        if self.on_progress:
            self.on_progress(stage, progress)

    def load(self, file_path: str) -> np.ndarray:
        self._report("Loading audio...", 0.0)
        self.audio_data, self.sample_rate = load_audio(file_path)
        self.pitch_result = None
        self.notes = None
        self.midi_file = None
        self._report("Audio loaded", 1.0)
        return self.audio_data

    def run_pitch_detection(
        self,
        fmin: float = 65.0,
        fmax: float = 2093.0,
        hop_length: int = 256,
    ) -> PitchResult:
        if self.audio_data is None:
            raise RuntimeError("No audio loaded")

        self._report("Detecting pitch...", 0.0)
        self.pitch_result = detect_pitch(
            self.audio_data, self.sample_rate,
            fmin=fmin, fmax=fmax, hop_length=hop_length,
        )
        self.notes = None
        self.midi_file = None
        self._report("Pitch detected", 1.0)
        return self.pitch_result

    def run_note_segmentation(
        self,
        min_note_duration: float = 0.05,
        pitch_stability_threshold: float = 1.5,
        voiced_threshold: float = 0.4,
    ) -> List[NoteEvent]:
        if self.pitch_result is None:
            raise RuntimeError("No pitch data")

        self._report("Segmenting notes...", 0.0)
        self.notes = segment_notes(
            self.pitch_result,
            self.audio_data,
            self.sample_rate,
            min_note_duration=min_note_duration,
            pitch_stability_threshold=pitch_stability_threshold,
            voiced_threshold=voiced_threshold,
        )
        self.midi_file = None
        self._report("Notes segmented", 1.0)
        return self.notes

    def run_midi_generation(
        self,
        tempo: int = 120,
        instrument: int = 0,
        default_velocity: Optional[int] = None,
        scale_filter: Optional[Tuple[str, str]] = None,
        pitch_bend: bool = True,
    ) -> MIDIFile:
        if self.notes is None:
            raise RuntimeError("No notes segmented")

        self._report("Generating MIDI...", 0.0)
        self.midi_file = generate_midi(
            self.notes,
            tempo=tempo,
            instrument=instrument,
            default_velocity=default_velocity,
            scale_filter=scale_filter,
            pitch_bend=pitch_bend,
        )
        self._report("MIDI generated", 1.0)
        return self.midi_file

    def export_midi(self, output_path: str) -> None:
        if self.midi_file is None:
            raise RuntimeError("No MIDI generated")
        save_midi(self.midi_file, output_path)

    def render_preview(self) -> np.ndarray:
        if self.notes is None:
            raise RuntimeError("No notes to preview")
        return render_notes_to_audio(self.notes)

    def run_full_pipeline(
        self,
        file_path: str,
        tempo: int = 120,
        instrument: int = 0,
        min_note_duration: float = 0.05,
        voiced_threshold: float = 0.5,
        scale_filter: Optional[Tuple[str, str]] = None,
    ) -> MIDIFile:
        self.load(file_path)
        self.run_pitch_detection()
        self.run_note_segmentation(
            min_note_duration=min_note_duration,
            voiced_threshold=voiced_threshold,
        )
        return self.run_midi_generation(
            tempo=tempo,
            instrument=instrument,
            scale_filter=scale_filter,
        )
