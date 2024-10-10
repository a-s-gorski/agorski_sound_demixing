import pytest
import torch
from pipelines.utils.loss import SDRLoss, get_loss, WaveformLoss, SpectrogramLoss
from pipelines.types.training_waveform import TrainingWaveformConfig
from pipelines.types.training_spectrogram import TrainingSpectrogramConfig

@pytest.fixture
def waveform_config() -> TrainingWaveformConfig:
    config = TrainingWaveformConfig(
        model="convtasnet",
        loss= "mse",
        sources=["bass.wav"],
        num_channels=1
    )
    return config

@pytest.fixture
def spectrogram_config() -> TrainingSpectrogramConfig:
    config = TrainingSpectrogramConfig(
        model="unet",
        loss="mse",
        input_channels=2,
        sources=["bass.wav"],
    )
    return config

def test_sdrloss():
    sdr_loss = SDRLoss()

    # Test case 1: two empty signals should results in zero loss
    y_true = torch.zeros(10)
    y_pred = torch.zeros(10)
    loss = sdr_loss(y_true, y_pred)
    assert loss == 0

    # Test case 2: two random signals should have non-zero positive, finite loss
    y_true = torch.randn(100)
    y_pred = torch.randn(100)
    loss = sdr_loss(y_true, y_pred)
    assert torch.isfinite(loss) and loss.item() > 0

    # Test case 3: testing numerical stability for signals with large differences
    y_true = torch.tensor([1e10])
    y_pred = torch.tensor([1.00001e10])
    loss = sdr_loss(y_true, y_pred)
    assert torch.isfinite(loss) and loss.item() >= 0


def test_get_loss_waveform_mse(waveform_config):
    loss_fn = get_loss(waveform_config)
    assert isinstance(loss_fn, torch.nn.MSELoss)


def test_get_loss_waveform_mae(waveform_config):
    waveform_config.loss = WaveformLoss("mae")
    loss_fn = get_loss(waveform_config)
    assert isinstance(loss_fn, torch.nn.L1Loss)


def test_get_loss_waveform_sdr(waveform_config):
    waveform_config.loss = WaveformLoss("sdr")
    loss_fn = get_loss(waveform_config)
    assert isinstance(loss_fn, SDRLoss)


def test_get_loss_spectrogram_mse(spectrogram_config):
    loss_fn = get_loss(spectrogram_config)
    assert isinstance(loss_fn, torch.nn.MSELoss)


def test_get_loss_spectrogram_mae(spectrogram_config):
    spectrogram_config.loss = SpectrogramLoss("mae")
    loss_fn = get_loss(spectrogram_config)
    assert isinstance(loss_fn, torch.nn.L1Loss)


def test_get_loss_spectrogram_sdr(spectrogram_config):
    spectrogram_config.loss = SpectrogramLoss("sdr")
    loss_fn = get_loss(spectrogram_config)
    assert isinstance(loss_fn, SDRLoss)


if __name__ == "__main__":
    pytest.main()
