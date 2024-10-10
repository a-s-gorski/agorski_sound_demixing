from typing import Dict

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from tqdm import tqdm

from pipelines.dataset.base_dataset import BaseDataset
from pipelines.models.convtasnet import ConvTasNet
from pipelines.models.hdemucs import HDEMUCS
from pipelines.models.lstm import BiLSTM
from pipelines.types.training_signal_enriched import TrainingGenerativeConfig
from pipelines.types.training_spectrogram import TrainingSpectrogramConfig
from pipelines.types.training_waveform import (
    TrainingWaveformConfig,
    WaveformLoss,
    WaveformModel,
)
from pipelines.utils.loss import SDRLoss, get_loss


def get_model(config: TrainingWaveformConfig) -> BiLSTM | HDEMUCS | ConvTasNet:
    """
    Returns model based on training configuration.

    Parameters:
        config (TrainingWaveformConfig): training configuration

    Returns:
        pl.LightningModule: returns a pytorch-lightning based model for waveform source seperation training
    """

    loss = get_loss(config=config)
    match config.model:
        case WaveformModel.LSTM:
            model = BiLSTM(config=config, loss=loss)

        case WaveformModel.HDEMUCS:
            model = HDEMUCS(
                config=config, loss=loss
            )
        case WaveformModel.CONVTASNET:
            model = ConvTasNet(
                config=config, loss=loss
            )

    return model


def get_model_predictions(
        model: nn.Module,
        dataset: BaseDataset) -> torch.Tensor:
    """
    Generates model predictions for the given dataset using the provided neural network model.

    Parameters:
        model (nn.Module): The trained neural network model.
        dataset (BaseDataset): The dataset containing input audio waveforms.

    Returns:
        torch.Tensor: A tensor containing the model predictions for the input audio waveforms.
    """
    preds = list(
        tqdm(map(lambda x: model(x[0].unsqueeze(0),), dataset), total=len(dataset)))
    preds = torch.cat(preds, dim=0)
    return preds


def convert_to_stereo(dataset: BaseDataset) -> BaseDataset:
    """
    Helper function for converting signal to stereo, since some transfer learning models are trained on stereo

    Parameters:
        dataset(BaseDataset): both X and Y must have mono channels as a second last dimension

    Returns:
        torch.Tensor: reshaped signal with stereo channels
    """
    dataset.X = torch.cat([dataset.X, dataset.X], dim=-2)
    dataset.Y = torch.cat([dataset.Y, dataset.Y], dim=-2)
    return dataset


def convert_to_mono(signal: torch.Tensor) -> torch.Tensor:
    """
    Helper function for converting stereo signal back to mono, since some transfer learning models are trained on mono

    Parameters;
        signal (torch.Tensor): has to follow shape (num_batches, num_channels=2, sequence_length)

    Returns:
        torch.Tensor: output signal in mono with shape (num_batches, num_channels=1, sequence_length)
    """
    signal = list(map(librosa.to_mono, signal.numpy()))
    signal = list(map(lambda x: torch.from_numpy(x).unsqueeze(0), signal))
    signal = torch.cat(signal, dim=0)
    return signal


def convert_sources_signal_to_mono(signal: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around convert_to_mono for signal aggregated around seperated sources:

    Parameters:
        signal (torch.Tensor): has to follow shape (num_batches, num_sources, num_channels=2, sequence_length)

    Returns:
        torch.Tensor: will follow shape (num_batches, num_sources, num_channels=1, sequence_length)
    """
    # expected signal shape: (num_batches, num_sources, num_channels = 2,
    # signal_len)
    num_sources = signal.shape[1]
    signal = signal.reshape(-1, signal.shape[-2], signal.shape[-1])
    signal = convert_to_mono(signal=signal)
    signal = signal.reshape(-1, num_sources, 1, signal.shape[-1])
    return signal


def convert_dataset_to_mono(dataset: BaseDataset) -> BaseDataset:
    """
    Wrapper around convert_to_mono for BaseDataset object

    Parameters:
        dataset: X has to follow shape (num_batches, num_channals=1, seq_len), Y has to follow shape (num_batches, num_sources, num_channels=1, seq_len)

    Returns:
        torch.Tensor: will follow the shapes X (num_batches, num_channals=1, seq_len) Y (num_batches, num_sources, num_channels=1, seq_len)
    """
    # excepted x size : (num_batches, num_channels = 2, signal_len)
    # expected y size: (num_batches, num_sources, num_channels = 2, signal_len)

    dataset.X = convert_to_mono(dataset.X)
    dataset.X = dataset.X.reshape(-1, 1, dataset.X.shape[-1])
    dataset.Y = convert_sources_signal_to_mono(dataset.Y)

    return dataset


def calculate_mean_loss(
        loss: torch.Tensor,
        predicted: torch.Tensor,
        targets: torch.Tensor) -> float:
    """
    Returns mean loss across all sample batches.

    Parameters:
        loss (nn.Module) - loss function
        predicted (torch.Tensor) - seperated sources
        targets (torch.Tensor) - originally seperated sources

    """
    losses = [loss(p, t) for p, t in zip(predicted, targets)]
    mean_loss = np.mean(losses)
    return mean_loss.item()


def calculate_losses(config: TrainingWaveformConfig | TrainingSpectrogramConfig | TrainingGenerativeConfig,
                     predicted: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Helper function for getting dictionary with loss across and by sources.

    Parameters:
        config (TrainingWaveformConfig | TrainingSpectrogramConfig): file with sources specified
        predicted (torch.Tensor): predicted seperated sources
        targets (torch.Tensor): target seperated sources

    Returns:
        Dict[str, float] - keys represent metrics and values its results.
    """

    # calculate mean loss across sources
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    sdr_loss = SDRLoss()

    losses = {}

    losses["mae"] = calculate_mean_loss(mae_loss, predicted, targets)
    losses["mse"] = calculate_mean_loss(mse_loss, predicted, targets)
    losses["sdr"] = calculate_mean_loss(sdr_loss, predicted, targets)

    for source_index, source in enumerate(config.sources):
        source_name = source.removesuffix(".wav")

        predicted_source = predicted[:, source_index]
        target_source = targets[:, source_index]

        losses[f"{source_name}_mae"] = calculate_mean_loss(
            mae_loss, predicted_source, target_source)
        losses[f"{source_name}_mse"] = calculate_mean_loss(
            mse_loss, predicted_source, target_source)
        losses[f"{source_name}_sdr"] = calculate_mean_loss(
            sdr_loss, predicted_source, target_source)

    return losses


def get_signal_by_source(config: TrainingWaveformConfig | TrainingGenerativeConfig,
                         signal: torch.Tensor) -> Dict[str, NDArray]:
    """
    Helper function for getting signal by source.

    Parameters:
        config (TrainingWaveformConfig): parsed configuration file with sources names
        signal (torch.Tensor): seperated sources with shape (num_batches, num_sources, num_channels, subseq_len)
    """
    signal_by_source = {}
    for source_index, source in enumerate(config.sources):
        source_name = source.removeprefix(".wav")
        signal_by_source[source_name] = signal[:, source_index].numpy()
    return signal_by_source
