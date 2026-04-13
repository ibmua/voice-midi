import numpy as np
import librosa
from dataclasses import dataclass


@dataclass
class PitchResult:
    times: np.ndarray
    frequencies: np.ndarray
    voiced_flags: np.ndarray
    voiced_probs: np.ndarray
    hop_length: int = 256
    sr: int = 22050


def detect_pitch(
    samples: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 2093.0,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> PitchResult:
    """Ensemble pitch detection: combines Praat autocorrelation + librosa pyin
    for much more accurate and sensitive pitch tracking."""

    time_step = hop_length / sr

    # --- Detector 1: Praat autocorrelation (gold standard for speech/voice) ---
    praat_f0, praat_conf = _detect_praat(samples, sr, time_step, fmin, fmax)

    # --- Detector 2: librosa pyin (probabilistic YIN) ---
    pyin_f0, pyin_voiced, pyin_probs = _detect_pyin(
        samples, sr, fmin, fmax, frame_length, hop_length
    )

    # --- Detector 3: librosa yin (plain YIN, catches what pyin misses) ---
    yin_f0 = _detect_yin(samples, sr, fmin, fmax, frame_length, hop_length)

    # --- RMS energy for voicing decisions ---
    rms = librosa.feature.rms(
        y=samples, frame_length=frame_length, hop_length=hop_length
    )[0]

    # Align all to same length
    n_frames = min(len(pyin_f0), len(yin_f0), len(rms))
    if len(praat_f0) != n_frames:
        praat_f0 = _resample_to_length(praat_f0, n_frames)
        praat_conf = _resample_to_length(praat_conf, n_frames)
    pyin_f0 = pyin_f0[:n_frames]
    pyin_probs = pyin_probs[:n_frames]
    pyin_voiced = pyin_voiced[:n_frames]
    yin_f0 = yin_f0[:n_frames]
    rms = rms[:n_frames]

    times = librosa.times_like(pyin_f0[:n_frames], sr=sr, hop_length=hop_length)

    # --- Merge detectors ---
    f0, voiced, confidence = _merge_detectors(
        praat_f0, praat_conf,
        pyin_f0, pyin_probs, pyin_voiced,
        yin_f0, rms, fmin, fmax,
    )

    # Smooth in MIDI space
    voiced_mask = voiced & (f0 > 0)
    f0 = _smooth_pitch(f0, voiced_mask, kernel_size=5)

    return PitchResult(
        times=times[:n_frames],
        frequencies=f0,
        voiced_flags=voiced,
        voiced_probs=confidence,
        hop_length=hop_length,
        sr=sr,
    )


def _detect_praat(samples, sr, time_step, fmin, fmax):
    """Praat autocorrelation pitch detection — very accurate for voice."""
    try:
        import parselmouth

        snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch_ac(
            time_step=time_step,
            pitch_floor=fmin,
            pitch_ceiling=fmax,
            very_accurate=True,
        )

        freqs = pitch.selected_array['frequency']
        # Praat strength is per-candidate; approximate confidence from presence
        strength = np.array([
            pitch.get_strength_at(pitch.xs()[i]) if freqs[i] > 0 else 0.0
            for i in range(len(freqs))
        ])

        return freqs.astype(np.float64), strength.astype(np.float64)
    except Exception:
        return np.array([]), np.array([])


def _detect_pyin(samples, sr, fmin, fmax, frame_length, hop_length):
    """librosa pyin — probabilistic, good confidence estimates."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        samples, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    f0 = np.where(np.isnan(f0), 0.0, f0)
    voiced_flag = np.where(np.isnan(voiced_flag), False, voiced_flag).astype(bool)
    voiced_probs = np.where(np.isnan(voiced_probs), 0.0, voiced_probs)
    return f0, voiced_flag, voiced_probs


def _detect_yin(samples, sr, fmin, fmax, frame_length, hop_length):
    """librosa yin — deterministic, catches things pyin's voicing model rejects."""
    try:
        f0 = librosa.yin(
            samples, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=hop_length,
        )
        # YIN returns fmin for unvoiced frames; treat those as 0
        f0 = np.where((f0 <= fmin * 1.01) | (f0 >= fmax * 0.99), 0.0, f0)
        return f0
    except Exception:
        return np.array([])


def _resample_to_length(arr, target_len):
    """Resample an array to a target length via linear interpolation."""
    if len(arr) == 0:
        return np.zeros(target_len)
    if len(arr) == target_len:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def _merge_detectors(
    praat_f0, praat_conf,
    pyin_f0, pyin_probs, pyin_voiced,
    yin_f0, rms, fmin, fmax,
):
    """Combine multiple pitch detectors. Trust Praat most, then pyin, then yin.
    Use RMS energy to decide if a frame should be voiced at all."""

    n = len(pyin_f0)
    f0 = np.zeros(n, dtype=np.float64)
    voiced = np.zeros(n, dtype=bool)
    confidence = np.zeros(n, dtype=np.float64)

    # Compute noise floor from the quietest 20% of frames
    rms_sorted = np.sort(rms)
    noise_floor = rms_sorted[int(len(rms_sorted) * 0.2)] if len(rms_sorted) > 0 else 0.0
    energy_threshold = max(noise_floor * 2.0, 0.005)

    has_praat = len(praat_f0) == n

    for i in range(n):
        candidates = []

        # Praat result
        if has_praat and praat_f0[i] > 0 and fmin <= praat_f0[i] <= fmax:
            candidates.append((praat_f0[i], 0.85 + 0.15 * praat_conf[i], 'praat'))

        # pyin result
        if pyin_f0[i] > 0 and fmin <= pyin_f0[i] <= fmax:
            candidates.append((pyin_f0[i], pyin_probs[i], 'pyin'))

        # yin result (lower trust, but useful when others miss)
        if yin_f0[i] > 0 and fmin <= yin_f0[i] <= fmax:
            candidates.append((yin_f0[i], 0.5, 'yin'))

        if not candidates:
            continue

        # Need some energy to count as voiced
        if rms[i] < energy_threshold * 0.3:
            continue

        # If multiple detectors agree (within 1 semitone), boost confidence
        if len(candidates) >= 2:
            midi_vals = [69 + 12 * np.log2(c[0] / 440) for c in candidates]
            # Check agreement between top candidates
            best = max(candidates, key=lambda c: c[1])
            best_midi = 69 + 12 * np.log2(best[0] / 440)

            agreeing = sum(1 for mv in midi_vals if abs(mv - best_midi) < 1.0)
            agreement_bonus = 0.15 * (agreeing - 1)

            f0[i] = best[0]
            confidence[i] = min(1.0, best[1] + agreement_bonus)
            voiced[i] = True
        else:
            best = candidates[0]
            # Single detector: accept if has energy
            if rms[i] >= energy_threshold or best[1] > 0.6:
                f0[i] = best[0]
                confidence[i] = best[1]
                voiced[i] = True

    return f0, voiced, confidence


def _smooth_pitch(f0, voiced_mask, kernel_size=5):
    """Median-filter pitch in MIDI space within voiced runs."""
    from scipy.ndimage import median_filter

    result = f0.copy()
    if not np.any(voiced_mask):
        return result

    midi = np.zeros_like(f0)
    v = voiced_mask & (f0 > 0)
    midi[v] = 69.0 + 12.0 * np.log2(f0[v] / 440.0)

    changes = np.diff(v.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for s, e in zip(starts, ends):
        seg_len = e - s
        if seg_len >= kernel_size:
            midi[s:e] = median_filter(midi[s:e], size=kernel_size)
        elif seg_len >= 3:
            midi[s:e] = median_filter(midi[s:e], size=3)

    valid = midi > 0
    result[valid] = 440.0 * 2.0 ** ((midi[valid] - 69.0) / 12.0)
    result[~valid] = 0.0
    return result
