from pipelines.dataset.base_dataset import BaseDataset
from pipelines.models.convtasnet import ConvTasNet
from pipelines.models.hdemucs import HDEMUCS
from pipelines.models.lstm import BiLSTM

from pipelines.types.training_waveform import WaveformModel, WaveformLoss, TrainingWaveformConfig
from pipelines.utils.training_waveform import get_model, get_model_predictions, convert_to_stereo, convert_to_mono, convert_sources_signal_to_mono, convert_dataset_to_mono, calculate_mean_loss, calculate_losses, get_signal_by_source
import torch
import pytest
from typing import Tuple, List
import numpy as np

def test_get_model():
    config = TrainingWaveformConfig(
        model=WaveformModel.HDEMUCS,
        loss="mse",
        sources=["bass.wav", "vocals.wav"],
        num_channels=4
    )
    model = get_model(config)
    assert isinstance(model, HDEMUCS)

    config.model = WaveformModel.CONVTASNET
    model = get_model(config)
    assert isinstance(model, ConvTasNet)

    config.model = WaveformModel.LSTM
    model = get_model(config)
    assert isinstance(model, BiLSTM)


@pytest.mark.parametrize(
        "dataset,expected_shape_x,expected_shape_y",
        [
            (BaseDataset(X=torch.randn(2,1,10000),Y=torch.randn(2,4,1,10000)), [2,2,10000], [2,4,2,10000]),
            (BaseDataset(X=torch.randn(1,100),Y=torch.randn(4,1,100)), [2,100], [4,2,100])
        ]
)
def test_convert_to_stereo(dataset: BaseDataset, expected_shape_x: Tuple[int], expected_shape_y: Tuple[int]):
    processed_ds = convert_to_stereo(dataset)
    assert processed_ds.X.size() == torch.Size(expected_shape_x) and processed_ds.Y.size() == torch.Size(expected_shape_y)


@pytest.mark.parametrize(
        "dataset,expected_shape",
        [
            (torch.randn(2,2,10000), [2,10000]),
            (torch.randn(10,2,100), [10,100]),
        ]
)
def test_convert_to_mono(dataset: BaseDataset, expected_shape: Tuple[int]):
    processed_signal = convert_to_mono(dataset)
    assert processed_signal.shape == torch.Size(expected_shape)

@pytest.fixture
def mock_predicted_targets():
    # Mock predicted and target sources for testing
    batch_size = 5
    num_sources = 3
    num_channels = 2
    subseq_len = 100

    predicted = torch.rand(batch_size, num_sources, num_channels, subseq_len)
    targets = torch.rand(batch_size, num_sources, num_channels, subseq_len)

    return predicted, targets


@pytest.fixture
def mock_config():
    class MockConfig:
        def __init__(self):
            self.sources = [".source1.wav", ".source2.wav", ".source3.wav"]

    return MockConfig()


def test_calculate_mean_loss(mock_predicted_targets):
    predicted, targets = mock_predicted_targets
    loss_module = torch.nn.MSELoss()
    mean_loss = calculate_mean_loss(loss_module, predicted, targets)
    assert isinstance(mean_loss, float)


def test_calculate_losses(mock_predicted_targets, mock_config):
    predicted, targets = mock_predicted_targets
    losses = calculate_losses(mock_config, predicted, targets)
    assert isinstance(losses, dict)
    assert "mae" in losses
    assert "mse" in losses
    assert "sdr" in losses

    # Check per-source losses
    for source in mock_config.sources:
        source_name = source.removesuffix(".wav")
        assert f"{source_name}_mae" in losses
        assert f"{source_name}_mse" in losses
        assert f"{source_name}_sdr" in losses


def test_get_signal_by_source(mock_predicted_targets, mock_config):
    predicted, _ = mock_predicted_targets
    signal_by_source = get_signal_by_source(mock_config, predicted)
    assert isinstance(signal_by_source, dict)

    for source in mock_config.sources:
        source_name = source.removeprefix(".wav")
        assert source_name in signal_by_source

    for source_signal in signal_by_source.values():
        assert isinstance(source_signal, np.ndarray)
        assert source_signal.shape == (predicted.shape[0], predicted.shape[2], predicted.shape[3])


if __name__ == "__main__":
    pytest.main()
