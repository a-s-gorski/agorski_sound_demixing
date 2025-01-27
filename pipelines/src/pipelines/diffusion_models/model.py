import pytorch_lightning as pl
import torch
from audio_diffusion_pytorch import AudioDiffusionModel


class Model(pl.LightningModule):
    """
    PyTorch Lightning module for training an Audio Diffusion Model.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Beta1 parameter for Adam optimizer.
        beta2 (float): Beta2 parameter for Adam optimizer.
        *args: Additional arguments for the AudioDiffusionModel.
        **kwargs: Additional keyword arguments for the AudioDiffusionModel.
    """

    def __init__(
        self, learning_rate: float, beta1: float, beta2: float, *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = AudioDiffusionModel(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model(waveforms)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model(waveforms)
        self.log("valid_loss", loss)
        return loss
