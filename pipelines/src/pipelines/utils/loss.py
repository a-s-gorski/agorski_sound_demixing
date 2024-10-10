import torch
from torch import nn

from pipelines.types.training_spectrogram import (
    SpectrogramLoss,
    TrainingSpectrogramConfig,
)
from pipelines.types.training_waveform import TrainingWaveformConfig, WaveformLoss


class SDRLoss(nn.Module):
    """
    Custom implementation of signal to distortion loss function used in Music Source Seperation Competition

    Methods:
        forward(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor
            Calculates loss function based on two signal, real one and the estimated one.

    Usage Example:
        sdr_loss = SDRLoss()
        predicted, estimated = torch.randn(100, 1), torch.randn(100, 1)
        loss = sdr_loss(predicted, estimated)

    """

    def __init__(self) -> None:
        super(SDRLoss, self).__init__()

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss based on power of the original and the estimated signals.
        Both signals must have exactly the same shapes.

        Parameters:
            y_true (torch.Tensor): The original signal
            t_pred (torch.Tensor): The estimated signal

        Returns:
            torch.Tensor: The ratio of the true signal power to distortion
        """
        delta = 1e-7  # added for numeric stability
        y_true_pow = torch.mean(y_true ** 2)
        distortion = y_true - y_pred
        distortion = torch.mean(distortion ** 2)
        y_true_pow += delta
        distortion += delta
        return torch.abs(10 * torch.log10(y_true_pow / distortion))


def get_loss(config: TrainingWaveformConfig |
             TrainingSpectrogramConfig) -> nn.Module:
    """
    Helper function to get loss function for training.

    Parameters:
        config (TrainingWaveformConfig | TrainingSpectrogramConfig): Parsed configuration from parameters.yml with loss function type specified

    Returns:
        nn.Module: returns an object which can be used as a loss function for model training.
    """
    if isinstance(config, TrainingWaveformConfig):
        match config.loss:
            case WaveformLoss.MSE:
                return nn.MSELoss()
            case WaveformLoss.MAE:
                return nn.L1Loss()
            case WaveformLoss.SDR:
                return SDRLoss()
    elif isinstance(config, TrainingSpectrogramConfig):
        match config.loss:
            case SpectrogramLoss.MSE:
                return nn.MSELoss()
            case SpectrogramLoss.MAE:
                return nn.L1Loss()
            case SpectrogramLoss.SDR:
                return SDRLoss()
