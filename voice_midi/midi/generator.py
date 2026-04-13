import io
import numpy as np
from typing import List, Optional, Tuple
from midiutil import MIDIFile
from ..analysis.music_utils import NoteEvent, filter_to_scale


def generate_midi(
    notes: List[NoteEvent],
    tempo: int = 120,
    instrument: int = 0,
    channel: int = 0,
    default_velocity: Optional[int] = None,
    quantize_pitch: bool = True,
    scale_filter: Optional[Tuple[str, str]] = None,
    pitch_bend: bool = True,
    bend_range: int = 2,
) -> MIDIFile:
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    midi.addProgramChange(0, channel, 0, instrument)

    # Set pitch bend range via RPN
    if pitch_bend:
        _set_pitch_bend_range(midi, channel, bend_range)

    beats_per_second = tempo / 60.0

    for note in notes:
        midi_note = note.midi_note

        if scale_filter:
            root, scale = scale_filter
            midi_note = filter_to_scale(midi_note, root, scale)

        midi_note = max(0, min(127, midi_note))
        velocity = default_velocity if default_velocity else note.velocity

        start_beats = note.start_time * beats_per_second
        duration_beats = max(note.duration * beats_per_second, 0.01)

        midi.addNote(0, channel, midi_note, start_beats, duration_beats, velocity)

        # Add pitch bend curve for actual pitch
        if pitch_bend and note.pitch_curve:
            _add_pitch_bends(
                midi, channel, note, midi_note, beats_per_second, bend_range
            )

    return midi


def _set_pitch_bend_range(midi, channel, semitones):
    """Set pitch bend range via RPN 0x0000."""
    midi.addControllerEvent(0, channel, 0, 101, 0)
    midi.addControllerEvent(0, channel, 0, 100, 0)
    midi.addControllerEvent(0, channel, 0, 6, semitones)
    midi.addControllerEvent(0, channel, 0, 38, 0)
    midi.addControllerEvent(0, channel, 0, 101, 127)
    midi.addControllerEvent(0, channel, 0, 100, 127)


def _add_pitch_bends(midi, channel, note, base_midi, beats_per_second, bend_range):
    """Add pitch bend events from a note's pitch curve."""
    curve = note.pitch_curve
    if not curve:
        return

    # Limit to ~30 events per second and skip tiny changes
    min_dt = 1.0 / 30.0
    last_t = -999.0
    last_bend = 0

    for t, freq in curve:
        if freq <= 0:
            continue
        if t - last_t < min_dt:
            continue

        actual_midi = 69.0 + 12.0 * np.log2(freq / 440.0)
        deviation = actual_midi - base_midi
        deviation = max(-bend_range, min(bend_range, deviation))

        bend_val = int(deviation / bend_range * 8192)
        bend_val = max(-8192, min(8191, bend_val))

        if abs(bend_val - last_bend) < 40:
            continue

        midi.addPitchWheelEvent(0, channel, t * beats_per_second, bend_val)
        last_t = t
        last_bend = bend_val

    # Reset pitch bend at end of note
    midi.addPitchWheelEvent(0, channel, note.end_time * beats_per_second, 0)


def save_midi(midi_file: MIDIFile, output_path: str) -> None:
    with open(output_path, 'wb') as f:
        midi_file.writeFile(f)


def get_midi_bytes(midi_file: MIDIFile) -> bytes:
    buf = io.BytesIO()
    midi_file.writeFile(buf)
    return buf.getvalue()
