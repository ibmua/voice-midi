import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class NoteEvent:
    start_time: float
    end_time: float
    duration: float
    frequency: float
    midi_note: int
    velocity: int
    confidence: float
    pitch_curve: list = field(default_factory=list)  # [(time, freq), ...]


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALE_INTERVALS = {
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
    'blues': [0, 3, 5, 6, 7, 10],
}

GM_INSTRUMENTS = {
    'Acoustic Grand Piano': 0,
    'Bright Acoustic Piano': 1,
    'Electric Grand Piano': 2,
    'Honky-tonk Piano': 3,
    'Electric Piano 1': 4,
    'Electric Piano 2': 5,
    'Harpsichord': 6,
    'Clavinet': 7,
    'Celesta': 8,
    'Glockenspiel': 9,
    'Music Box': 10,
    'Vibraphone': 11,
    'Marimba': 12,
    'Xylophone': 13,
    'Tubular Bells': 14,
    'Dulcimer': 15,
    'Drawbar Organ': 16,
    'Percussive Organ': 17,
    'Rock Organ': 18,
    'Church Organ': 19,
    'Reed Organ': 20,
    'Accordion': 21,
    'Harmonica': 22,
    'Tango Accordion': 23,
    'Acoustic Guitar (nylon)': 24,
    'Acoustic Guitar (steel)': 25,
    'Electric Guitar (jazz)': 26,
    'Electric Guitar (clean)': 27,
    'Electric Guitar (muted)': 28,
    'Overdriven Guitar': 29,
    'Distortion Guitar': 30,
    'Guitar Harmonics': 31,
    'Acoustic Bass': 32,
    'Electric Bass (finger)': 33,
    'Electric Bass (pick)': 34,
    'Fretless Bass': 35,
    'Slap Bass 1': 36,
    'Slap Bass 2': 37,
    'Synth Bass 1': 38,
    'Synth Bass 2': 39,
    'Violin': 40,
    'Viola': 41,
    'Cello': 42,
    'Contrabass': 43,
    'Tremolo Strings': 44,
    'Pizzicato Strings': 45,
    'Orchestral Harp': 46,
    'Timpani': 47,
    'String Ensemble 1': 48,
    'String Ensemble 2': 49,
    'Synth Strings 1': 50,
    'Synth Strings 2': 51,
    'Choir Aahs': 52,
    'Voice Oohs': 53,
    'Synth Choir': 54,
    'Orchestra Hit': 55,
    'Trumpet': 56,
    'Trombone': 57,
    'Tuba': 58,
    'Muted Trumpet': 59,
    'French Horn': 60,
    'Brass Section': 61,
    'Synth Brass 1': 62,
    'Synth Brass 2': 63,
    'Soprano Sax': 64,
    'Alto Sax': 65,
    'Tenor Sax': 66,
    'Baritone Sax': 67,
    'Oboe': 68,
    'English Horn': 69,
    'Bassoon': 70,
    'Clarinet': 71,
    'Piccolo': 72,
    'Flute': 73,
    'Recorder': 74,
    'Pan Flute': 75,
    'Blown Bottle': 76,
    'Shakuhachi': 77,
    'Whistle': 78,
    'Ocarina': 79,
    'Lead 1 (square)': 80,
    'Lead 2 (sawtooth)': 81,
    'Lead 3 (calliope)': 82,
    'Lead 4 (chiff)': 83,
    'Lead 5 (charang)': 84,
    'Lead 6 (voice)': 85,
    'Lead 7 (fifths)': 86,
    'Lead 8 (bass + lead)': 87,
    'Pad 1 (new age)': 88,
    'Pad 2 (warm)': 89,
    'Pad 3 (polysynth)': 90,
    'Pad 4 (choir)': 91,
    'Pad 5 (bowed)': 92,
    'Pad 6 (metallic)': 93,
    'Pad 7 (halo)': 94,
    'Pad 8 (sweep)': 95,
}


def frequency_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_to_frequency(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def quantize_to_semitone(midi_float: float) -> int:
    return int(round(midi_float))


def midi_to_note_name(midi_note: int) -> str:
    octave = (midi_note // 12) - 1
    note = NOTE_NAMES[midi_note % 12]
    return f"{note}{octave}"


def note_name_to_midi(name: str) -> int:
    for i, n in enumerate(NOTE_NAMES):
        if name.upper().startswith(n):
            rest = name[len(n):]
            if rest.lstrip('-').isdigit():
                octave = int(rest)
                return (octave + 1) * 12 + i
    raise ValueError(f"Invalid note name: {name}")


def filter_to_scale(midi_note: int, root: str, scale: str) -> int:
    if scale not in SCALE_INTERVALS:
        return midi_note

    root_idx = NOTE_NAMES.index(root) if root in NOTE_NAMES else 0
    intervals = SCALE_INTERVALS[scale]
    scale_notes = [(root_idx + i) % 12 for i in intervals]

    note_class = midi_note % 12
    if note_class in scale_notes:
        return midi_note

    best_dist = 12
    best_note = midi_note
    for sn in scale_notes:
        dist_up = (sn - note_class) % 12
        dist_down = (note_class - sn) % 12
        dist = min(dist_up, dist_down)
        if dist < best_dist:
            best_dist = dist
            if dist_up <= dist_down:
                best_note = midi_note + dist_up
            else:
                best_note = midi_note - dist_down

    return max(0, min(127, best_note))
