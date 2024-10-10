import torch
import torch.nn.functional as F
import numpy as np
from pipelines.types.prepare_spectrogram import SpectrogramConfig
from pipelines.utils.process_spectrogram import compute_spectrogram
import pytest

def test_compute_spectrogram_no_masks_no_db():
    config = SpectrogramConfig(sample_freq=44100, n_fft=1024, time_mask=False, freq_mask=False, to_db=False,
                               output_width=512, output_height=512, time_mask_param=100, freq_mask_param=10)
    signal = torch.randn(1, 1, 44100)  # Random signal of shape (num_batches, num_channels, subsequence_shape)
    spectrogram, original_sizes = compute_spectrogram(signal, config)

    assert spectrogram.shape == torch.Size([1, 1, 512, 512])

    assert original_sizes == (513, 87)

def test_compute_spectrogram_with_masks_no_db():
    config = SpectrogramConfig(sample_freq=44100, n_fft=1024, time_mask=True, freq_mask=True, to_db=False,
                               output_width=512, output_height=512, time_mask_param=100, freq_mask_param=10)
    signal = torch.randn(1, 1, 44100)  # Random signal of shape (num_batches, num_channels, subsequence_shape)
    spectrogram, original_sizes = compute_spectrogram(signal, config)

    assert spectrogram.shape == torch.Size([1, 1, 512, 512])

    assert original_sizes == (513, 87)

def test_compute_spectrogram_with_db():
    # Test the function with converting to dB
    config = SpectrogramConfig(sample_freq=44100, n_fft=1024, time_mask=False, freq_mask=False, to_db=True,
                               output_width=512, output_height=512, time_mask_param=100, freq_mask_param=10)
    signal = torch.randn(1, 1, 44100)  # Random signal of shape (num_batches, num_channels, subsequence_shape)
    spectrogram, original_sizes = compute_spectrogram(signal, config)

    assert spectrogram.shape == torch.Size([1, 1, 512, 512])

    assert original_sizes == (513, 87)

    assert torch.isfinite(spectrogram).all()  # Check for finite values

if __name__ == "__main__":
    pytest.main()
