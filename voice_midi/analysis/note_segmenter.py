import numpy as np
import librosa
from typing import List
from .pitch_detector import PitchResult
from .music_utils import NoteEvent, frequency_to_midi, quantize_to_semitone


def segment_notes(
    pitch_result: PitchResult,
    samples: np.ndarray,
    sr: int,
    min_note_duration: float = 0.05,
    pitch_stability_threshold: float = 1.5,
    voiced_threshold: float = 0.4,
) -> List[NoteEvent]:
    times = pitch_result.times
    freqs = pitch_result.frequencies
    voiced = pitch_result.voiced_flags
    probs = pitch_result.voiced_probs
    hop_length = getattr(pitch_result, 'hop_length', 256)

    # Build voiced mask — be generous to capture anything note-like
    mask = voiced & (probs >= voiced_threshold) & (freqs > 0)

    # Also include frames where f0 is detected even with lower confidence
    # This catches quieter passages that pyin is less sure about
    soft_mask = (freqs > 0) & (probs >= voiced_threshold * 0.5)
    # Expand voiced regions slightly: if a soft-voiced frame is adjacent to
    # a confident frame, include it
    expanded = mask.copy()
    for i in range(1, len(mask) - 1):
        if soft_mask[i] and (mask[i - 1] or mask[i + 1]):
            expanded[i] = True
    mask = expanded

    # Detect onsets for note boundary hints
    onset_frames = _detect_onsets(samples, sr, hop_length)
    onset_set = set(onset_frames.tolist()) if len(onset_frames) > 0 else set()

    # Onset strength envelope for velocity estimation
    onset_env = _onset_strength(samples, sr, hop_length)

    # Find voiced runs
    voiced_runs = _find_runs(mask)

    notes = []
    for run_start, run_end in voiced_runs:
        # Find onsets within this run (not at the very start)
        run_onsets = sorted([f for f in onset_set if run_start < f < run_end])

        # Split points: start of run, each onset, end of run
        boundaries = [run_start] + run_onsets + [run_end]

        for b in range(len(boundaries) - 1):
            seg_start = boundaries[b]
            seg_end = boundaries[b + 1]

            # Further split on large pitch jumps
            sub_segments = _split_on_pitch_jumps(
                seg_start, seg_end, freqs, pitch_stability_threshold
            )

            for ss, se in sub_segments:
                note = _make_note(
                    ss, se, times, freqs, probs, samples, sr, onset_env
                )
                if note and note.duration >= min_note_duration:
                    notes.append(note)

    # Merge very short notes into neighbors if pitch is close
    notes = _merge_short_notes(notes, min_duration=min_note_duration * 0.8)

    return notes


def _detect_onsets(samples, sr, hop_length):
    """Detect note onsets using spectral flux."""
    try:
        onset_env = librosa.onset.onset_strength(
            y=samples, sr=sr, hop_length=hop_length
        )
        onsets = librosa.onset.onset_detect(
            y=samples, sr=sr, hop_length=hop_length,
            onset_envelope=onset_env,
            backtrack=True,
        )
        return onsets
    except Exception:
        return np.array([], dtype=int)


def _onset_strength(samples, sr, hop_length):
    """Get onset strength envelope for velocity estimation."""
    try:
        return librosa.onset.onset_strength(
            y=samples, sr=sr, hop_length=hop_length
        )
    except Exception:
        return None


def _find_runs(mask):
    """Find contiguous True runs in a boolean mask."""
    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _split_on_pitch_jumps(seg_start, seg_end, freqs, threshold):
    """Split a segment where pitch jumps more than threshold semitones."""
    if seg_end - seg_start < 2:
        return [(seg_start, seg_end)]

    seg_freqs = freqs[seg_start:seg_end]
    midi = np.array([frequency_to_midi(f) if f > 0 else 0 for f in seg_freqs])

    splits = [0]
    for j in range(1, len(midi)):
        if midi[j] > 0 and midi[j - 1] > 0:
            if abs(midi[j] - midi[j - 1]) > threshold:
                splits.append(j)
    splits.append(len(midi))

    result = []
    for k in range(len(splits) - 1):
        s = seg_start + splits[k]
        e = seg_start + splits[k + 1]
        if e > s:
            result.append((s, e))
    return result


def _make_note(seg_start, seg_end, times, freqs, probs, samples, sr, onset_env):
    """Create a NoteEvent from a frame range, preserving the pitch curve."""
    if seg_end <= seg_start + 1:
        return None

    seg_end = min(seg_end, len(times))
    if seg_start >= seg_end:
        return None

    start_time = times[seg_start]
    if seg_end < len(times):
        end_time = times[seg_end]
    else:
        end_time = times[-1]

    duration = end_time - start_time
    if duration <= 0:
        return None

    note_freqs = freqs[seg_start:seg_end]
    valid_freqs = note_freqs[note_freqs > 0]
    if len(valid_freqs) == 0:
        return None

    median_freq = float(np.median(valid_freqs))
    midi_float = frequency_to_midi(median_freq)
    midi_note = quantize_to_semitone(midi_float)
    midi_note = max(0, min(127, midi_note))

    # Velocity from RMS, boosted by onset strength
    sample_start = int(start_time * sr)
    sample_end = int(end_time * sr)
    sample_start = max(0, min(sample_start, len(samples) - 1))
    sample_end = max(sample_start + 1, min(sample_end, len(samples)))
    rms = np.sqrt(np.mean(samples[sample_start:sample_end] ** 2))

    if onset_env is not None and seg_start < len(onset_env):
        region = onset_env[seg_start:min(seg_start + 4, len(onset_env))]
        if len(region) > 0:
            rms = rms * (1.0 + float(np.max(region)) * 0.3)

    velocity = int(np.clip(rms * 400, 30, 127))
    confidence = float(np.mean(probs[seg_start:seg_end]))

    # Build pitch curve: list of (time, frequency) pairs
    pitch_curve = []
    for i in range(seg_start, seg_end):
        if i < len(times) and freqs[i] > 0:
            pitch_curve.append((float(times[i]), float(freqs[i])))

    return NoteEvent(
        start_time=float(start_time),
        end_time=float(end_time),
        duration=float(duration),
        frequency=median_freq,
        midi_note=midi_note,
        velocity=velocity,
        confidence=confidence,
        pitch_curve=pitch_curve,
    )


def _merge_short_notes(notes, min_duration=0.04):
    """Merge very short notes into adjacent notes with similar pitch."""
    if len(notes) < 2:
        return notes

    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        # Merge if both are short and pitch is close (within 2 semitones)
        if (note.duration < min_duration and prev.duration < min_duration
                and abs(note.midi_note - prev.midi_note) <= 2):
            # Combine into one note
            combined_curve = prev.pitch_curve + note.pitch_curve
            all_freqs = [f for _, f in combined_curve if f > 0]
            med_freq = float(np.median(all_freqs)) if all_freqs else prev.frequency
            midi_float = frequency_to_midi(med_freq)
            midi_note = max(0, min(127, quantize_to_semitone(midi_float)))

            merged[-1] = NoteEvent(
                start_time=prev.start_time,
                end_time=note.end_time,
                duration=note.end_time - prev.start_time,
                frequency=med_freq,
                midi_note=midi_note,
                velocity=max(prev.velocity, note.velocity),
                confidence=(prev.confidence + note.confidence) / 2,
                pitch_curve=combined_curve,
            )
        else:
            merged.append(note)
    return merged
