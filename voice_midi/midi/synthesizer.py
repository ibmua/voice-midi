import numpy as np
from typing import List, Optional
from ..analysis.music_utils import NoteEvent, midi_to_frequency


def render_notes_to_audio(
    notes: List[NoteEvent],
    sample_rate: int = 44100,
    total_duration: Optional[float] = None,
) -> np.ndarray:

    if not notes:
        return np.zeros(sample_rate, dtype=np.float32)

    if total_duration is None:
        total_duration = max(n.end_time for n in notes) + 0.5

    num_samples = int(total_duration * sample_rate)
    output = np.zeros(num_samples, dtype=np.float64)

    for note in notes:
        freq = midi_to_frequency(note.midi_note)
        amplitude = note.velocity / 127.0

        start_sample = int(note.start_time * sample_rate)
        end_sample = int(note.end_time * sample_rate)
        start_sample = max(0, min(start_sample, num_samples))
        end_sample = max(start_sample, min(end_sample, num_samples))

        n_samples = end_sample - start_sample
        if n_samples <= 0:
            continue

        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        tone = np.sin(2 * np.pi * freq * t) * amplitude

        # ADSR envelope
        envelope = _adsr_envelope(n_samples, sample_rate)
        tone *= envelope

        output[start_sample:end_sample] += tone

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.8

    return output.astype(np.float32)


def _adsr_envelope(
    n_samples: int,
    sample_rate: int,
    attack: float = 0.01,
    decay: float = 0.05,
    sustain_level: float = 0.7,
    release: float = 0.05,
) -> np.ndarray:
    envelope = np.ones(n_samples, dtype=np.float64)

    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    # Attack
    if attack_samples > 0:
        end = min(attack_samples, n_samples)
        envelope[:end] = np.linspace(0, 1, end)

    # Decay
    decay_start = attack_samples
    decay_end = min(decay_start + decay_samples, n_samples)
    if decay_end > decay_start:
        envelope[decay_start:decay_end] = np.linspace(1, sustain_level, decay_end - decay_start)

    # Sustain
    sustain_start = decay_end
    release_start = max(n_samples - release_samples, sustain_start)
    if release_start > sustain_start:
        envelope[sustain_start:release_start] = sustain_level

    # Release
    if release_start < n_samples:
        envelope[release_start:] = np.linspace(sustain_level, 0, n_samples - release_start)

    return envelope
