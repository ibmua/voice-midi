import os
import numpy as np
from typing import Tuple

SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
SOUNDFILE_FORMATS = {'.wav', '.flac'}


def validate_file(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_FORMATS


def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}")

    if ext in SOUNDFILE_FORMATS:
        return _load_with_soundfile(file_path, sr)
    else:
        return _load_with_pydub(file_path, sr)


def _load_with_soundfile(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    import librosa

    samples, file_sr = sf.read(file_path, dtype='float32', always_2d=True)
    samples = samples.mean(axis=1)  # mono

    if file_sr != target_sr:
        samples = librosa.resample(samples, orig_sr=file_sr, target_sr=target_sr)

    return samples, target_sr


def _load_with_pydub(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    from pydub import AudioSegment
    import librosa

    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # mono
    raw = np.array(audio.get_array_of_samples(), dtype=np.float32)
    raw /= np.iinfo(audio.array_type).max  # normalize to [-1, 1]

    file_sr = audio.frame_rate
    if file_sr != target_sr:
        raw = librosa.resample(raw, orig_sr=file_sr, target_sr=target_sr)

    return raw, target_sr


def get_duration(file_path: str) -> float:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SOUNDFILE_FORMATS:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    else:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0


def get_format_filter() -> list:
    return [
        ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
        ("WAV", "*.wav"),
        ("MP3", "*.mp3"),
        ("FLAC", "*.flac"),
        ("OGG", "*.ogg"),
        ("M4A", "*.m4a"),
        ("All Files", "*.*"),
    ]
