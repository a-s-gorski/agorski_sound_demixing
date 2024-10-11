import logging
import math
from typing import Tuple

import torch
from torch.nn import functional as F
from tqdm import tqdm

from pipelines.dataset.base_dataset import BaseDataset
from pipelines.models.convtasnet import ConvTasNet
from pipelines.models.gan import UnetGAN
from pipelines.models.hdemucs import HDEMUCS
from pipelines.models.lstm import BiLSTM
from pipelines.models.vae import VAE
from pipelines.types.training_signal_enriched import (
    GenerativeModel,
    TrainingGenerativeConfig,
)
from pipelines.utils.training import get_device
from pipelines.utils.training_waveform import convert_dataset_to_mono


def compute_enriched(model: BiLSTM | ConvTasNet | HDEMUCS,
                     dataset: BaseDataset) -> BaseDataset:
    """Compute the enriched dataset using the specified model.

    Args:
        model (BiLSTM | ConvTasNet | HDEMUCS): The model used to compute the enriched dataset.
        dataset (BaseDataset): The input dataset to be enriched.

    Returns:
        BaseDataset: The enriched dataset.
    """
    preds = list(tqdm(map(lambda x: model(x.detach().unsqueeze(0)),
                 dataset.X), total=len(dataset.X)))
    preds = torch.cat(preds, dim=0)
    dataset.X = preds.detach()
    dataset.Y = dataset.Y.detach()
    dataset.X.requires_grad = False
    dataset.Y.requires_grad = False
    return dataset


def signal_len_is_power_of_2(signal: torch.Tensor) -> bool:
    """Check if the length of the input signal is a power of 2.

    Args:
        signal (torch.Tensor): The input signal.

    Returns:
        bool: True if the signal length is a power of 2, False otherwise.
    """
    signal_len = signal.shape[-1]
    return signal_len > 0 and (signal_len & (signal_len - 1)) == 0


def pad_signal_to_power_of_2(signal: torch.Tensor) -> torch.Tensor:
    """Pad the input dataset to have a signal length that is a power of 2.

    Args:
        dataset (BaseDataset): The input dataset.

    Returns:
        BaseDataset: The padded dataset.
    """
    return F.pad(signal, (0, get_closest_larger_power_of_2(
        signal.shape[-1]) - signal.shape[-1]))


def get_closest_larger_power_of_2(signal_len: int) -> int:
    """Calculate the closest larger power of 2 for a given number.

    Args:
        signal_len (int): The input number.

    Returns:
        int: The closest larger power of 2.
    """
    return 2 ** math.ceil(math.log(signal_len, 2))


def pad_dataset(dataset: BaseDataset) -> BaseDataset:
    """Pad the input dataset to have a signal length that is a power of 2.

    Args:
        dataset (BaseDataset): The input dataset.

    Returns:
        BaseDataset: The padded dataset.
    """
    dataset.X = pad_signal_to_power_of_2(dataset.X)
    dataset.Y = pad_signal_to_power_of_2(dataset.Y)
    return dataset


def remove_mono_dim_dataset(
        dataset: BaseDataset,
        num_sources: int) -> BaseDataset:
    """Remove the mono dimension from the input dataset.

    Args:
        dataset (BaseDataset): The input dataset.
        num_sources (int): The number of audio sources in the dataset.

    Returns:
        BaseDataset: The dataset with the mono dimension removed.
    """
    dataset.X = remove_mono_dim_signal(dataset.X, num_sources)
    dataset.Y = remove_mono_dim_signal(dataset.Y, num_sources)
    return dataset


def remove_mono_dim_signal(
        signal: torch.Tensor,
        num_sources: int) -> torch.Tensor:
    """Remove the mono dimension from the input signal.

    Args:
        signal (torch.Tensor): The input signal.
        num_sources (int): The number of audio sources in the signal.

    Returns:
        torch.Tensor: The signal with the mono dimension removed.
    """
    signal = signal.reshape(signal.shape[0], num_sources, -1)
    return signal


def get_model(config: TrainingGenerativeConfig) -> UnetGAN | VAE:
    """Get the generative model based on the provided configuration.

    Args:
        config (TrainingGenerativeConfig): The configuration for training the generative model.

    Returns:
        UnetGAN | VAE: The selected generative model (UnetGAN or VAE).
    """
    match config.model:
        case GenerativeModel.GAN:
            device = get_device()
            return UnetGAN(config=config, device=device)
        case GenerativeModel.VAE:
            return VAE(config=config)


def prepare_model_input(config: TrainingGenerativeConfig,
                        dataset: BaseDataset) -> Tuple[BaseDataset,
                                                       TrainingGenerativeConfig,
                                                       int]:
    """Prepare the model input based on the provided configuration.

    Args:
        config (TrainingGenerativeConfig): The configuration for training the generative model.
        dataset (BaseDataset): The input dataset.

    Returns:
        Tuple[BaseDataset, TrainingGenerativeConfig, int]: The prepared dataset, updated configuration, and original signal length.
    """
    logger = logging.getLogger(__name__)

    original_signal_len = dataset.X.shape[-1]
    if not signal_len_is_power_of_2(
            dataset.X) and config.model == GenerativeModel.VAE:
        logger.warning(
            "Generative models expect signal length to be a power of 2 - padding")
        dataset = pad_dataset(dataset)
        config.signal_len = dataset.X.shape[-1]

    if len(dataset.X.shape) == 4 and dataset.X.shape[-2] == 2:
        logger.warning(
            "Generative models are trained on mono signal - converting from stero")
        dataset = convert_dataset_to_mono(dataset)

    if len(dataset.X.shape) == 4:
        dataset = remove_mono_dim_dataset(dataset, config.input_channels)

    return dataset, config, original_signal_len


def run_inference(
        model: VAE | UnetGAN,
        test_dataset: BaseDataset) -> torch.Tensor:
    """Run inference using the provided generative model on the test dataset.

    Args:
        model (VAE | UnetGAN): The trained generative model (VAE or UnetGAN).
        test_dataset (BaseDataset): The test dataset.

    Returns:
        torch.Tensor: The predictions obtained from the generative model.
    """
    if isinstance(model, VAE):
        predictions = list(tqdm(map(lambda x: model(
            x[0].unsqueeze(0))[-1], test_dataset), total=len(test_dataset)))
    else:
        predictions = list(tqdm(map(lambda x: model(
            x[0].unsqueeze(0)), test_dataset), total=len(test_dataset)))

    predictions = torch.cat(predictions, dim=0).detach()
    return predictions
