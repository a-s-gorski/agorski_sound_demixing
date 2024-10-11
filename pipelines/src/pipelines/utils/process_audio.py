import librosa
import numpy as np


def preprocess_audio(
    audio: np.array, mono: bool, origin_sr: float, sr: float, resample_type: str
) -> np.array:
    r"""Preprocess audio to mono / stereo, and resample.

    Args:
        audio: (channels_num, audio_samples), input audio
        mono: bool
        origin_sr: float, original sample rate
        sr: float, target sample rate
        resample_type: str, e.g., 'kaiser_fast'

    Returns:
        output: ndarray, output audio
    """
    if mono:
        audio = np.mean(audio, axis=0)
        # (audio_samples,)

    output = librosa.core.resample(
        audio, orig_sr=origin_sr, target_sr=sr, res_type=resample_type
    )
    # (audio_samples,) | (channels_num, audio_samples)

    if output.ndim == 1:
        output = output[None, :]
        # (1, audio_samples,)

    return output


def load_audio(
    audio_path: str,
    mono: bool,
    sample_rate: float,
    offset: float = 0.0,
    duration: float = None,
) -> np.array:
    r"""Load audio.

    Args:
        audio_path: str
        mono: bool
        sample_rate: float
    """
    audio, _ = librosa.core.load(
        audio_path, sr=sample_rate, mono=mono, offset=offset, duration=duration
    )
    # (audio_samples,) | (channels_num, audio_samples)

    if audio.ndim == 1:
        audio = audio[None, :]
        # (1, audio_samples,)

    return audio
