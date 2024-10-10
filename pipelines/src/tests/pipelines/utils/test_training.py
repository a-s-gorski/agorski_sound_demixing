import pytest
from unittest.mock import patch
from pipelines.utils.training import get_trainer, TrainingWaveformConfig, get_device
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
import torch

waveform_config = TrainingWaveformConfig(early_stopping=True, num_channels=1, sources=["bass.wav",], quality='loss', patience=3, save_checkpoint=True,
                                       model='lstm', loss="mse", checkpoint_output_path='/dummy_path', save_top_k=1, every_n_epochs=1, max_epochs=10, log_mlflow=True)


@pytest.fixture
def mock_mlflow():
    with patch('pipelines.utils.training.mlflow') as mock_mlflow:
        yield mock_mlflow

@pytest.fixture
def mock_os_makedirs():
    with patch('os.makedirs') as mock_makedirs:
        yield mock_makedirs

def test_get_trainer_with_waveform_config(mock_mlflow, mock_os_makedirs):
    trainer = get_trainer(waveform_config)

    assert isinstance(trainer, pl.Trainer)
    assert len(trainer.callbacks) == 4
    assert isinstance(trainer.callbacks[0], EarlyStopping)
    assert isinstance(trainer.logger, MLFlowLogger)
    assert trainer.log_every_n_steps == 1



def test_get_device():
    with patch('torch.cuda.is_available') as mock_is_available:
        mock_is_available.return_value = True
        assert get_device() == torch.device("cuda")

    with patch('torch.cuda.is_available') as mock_is_available:
        mock_is_available.return_value = False
        assert get_device() == torch.device("cpu")