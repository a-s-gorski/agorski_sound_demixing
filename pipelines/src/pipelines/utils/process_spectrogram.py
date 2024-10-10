from typing import Tuple

import librosa
import torch
from torch.nn import functional as F
from torchaudio import transforms as T

from pipelines.types.prepare_spectrogram import SpectrogramConfig


def compute_spectrogram(signal: torch.Tensor,
                        config: SpectrogramConfig) -> Tuple[torch.Tensor,
                                                            Tuple[float, float]]:
    """
    Wrapper of Torchaudio spectrogram transform with optional time/frequency masking.
    Signal is also padded to the shape defined in config (for unet spectrogram has to have each axis being power of 2)

    Parameters:
        signal (torch.Tensor): signal to be transformed with shape (num_batches, num_channels, subsequence_shape)
        config (SpectrogramConfig): parsed configuration file with specified transforms

    Returns:
        torch.Tensor: computed padded spectrograms
        Dict[str, int]: shape of the spectrogram before padding for reverse transform
    """
    spectrogram_transform = T.Spectrogram(n_fft=config.n_fft)
    freq_mask_transform = T.FrequencyMasking(
        freq_mask_param=config.freq_mask_param)
    time_mask_transform = T.TimeMasking(time_mask_param=config.time_mask_param)
    spectrogram = spectrogram_transform(signal)
    if config.freq_mask:
        spectrogram = freq_mask_transform(spectrogram)
    if config.time_mask:
        spectrogram = time_mask_transform(spectrogram)
    if config.to_db:
        spectrogram = spectrogram.numpy()
        spectrogram = librosa.amplitude_to_db(spectrogram)
        spectrogram = torch.from_numpy(spectrogram)

    original_sizes = (spectrogram.shape[-2], spectrogram.shape[-1])
    spectrogram = spectrogram[:, :,
                              :config.output_width, :config.output_height]
    spectrogram = F.pad(spectrogram,
                        (0,
                         config.output_width - spectrogram.shape[-1],
                         0,
                         config.output_height - spectrogram.shape[-2]))
    return spectrogram, original_sizes
