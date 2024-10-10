from typing import Dict

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchaudio import transforms as T
from tqdm import tqdm

from pipelines.models.unet import Unet
from pipelines.types.training_spectrogram import (
    SpectrogramModel,
    TrainingSpectrogramConfig,
)
from pipelines.utils.loss import get_loss
from pipelines.utils.training import get_trainer


def get_model(config: TrainingSpectrogramConfig) -> Unet:
    """
    Returns model based on training configuration.

    Parameters:
        config (TrainingSpectrogramConfig): training configuration

    Returns:
        pl.LightningModule: returns a pytorch-lightning based model for spectrogram segmentation training
    """

    loss = get_loss(config=config)

    match config.model:
        case SpectrogramModel.UNET:
            model = Unet(config=config, loss=loss)

    return model


def get_predictions(model: pl.LightningModule,
                    spectrograms: torch.Tensor) -> torch.Tensor:
    """
    Helper function for running inference:

    Parameters:
        model: pytorch-lightning compatible model
        spectrograms: spectrograms tensor with shape (num_batches, num_channels, height, width) both height and width have to be a power of 2

    Returns:
        torch.Tensor:

    """
    preds = list(tqdm(map(lambda x: model(x.unsqueeze(0)),
                 spectrograms), total=len(spectrograms)))
    preds = torch.cat(preds, dim=0)
    return preds


def convert_spectrogram_to_signal(
        spectrograms: torch.Tensor, orginal_shapes: Dict[str, int]):
    """
    Helper function for transforming spectrogram back to signal

    Parameters:
        spectrograms(torch.Tensor): output_spectrograms with shape (num_batches, num_sources, num_channels, spectrogram_height, spectrogram_height)

    """
    inverse_transform = T.InverseSpectrogram()
    w, h = orginal_shapes["width"], orginal_shapes["height"]
    spectrograms = spectrograms[:, :, :, :w, :h]
    spectrograms = F.pad(spectrograms,
                         (0,
                          h - spectrograms.shape[-1],
                          0,
                          w - spectrograms.shape[-2])).to(torch.cdouble)
    signal = inverse_transform(spectrograms)
    return signal
