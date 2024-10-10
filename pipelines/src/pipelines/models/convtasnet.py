from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX

from pipelines.types.training_waveform import TrainingWaveformConfig


class ConvTasNet(pl.LightningModule):
    """
    ConvTasNet: A PyTorch Lightning module implementing Conv-TasNet for source separation.

    Args:
        config (TrainingWaveformConfig): The configuration object containing training settings.
        loss (nn.Module): The loss function used for training.

    Attributes:
        models (nn.ModuleList): A list of Conv-TasNet models used for source separation.
        train_loss (float): Cumulative training loss during an epoch.
        val_loss (float): Cumulative validation loss during an epoch.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay (L2 regularization) for the optimizer.
        eps (float): Epsilon value for numerical stability in the optimizer.
        lr_scheduler (bool): Whether to use a learning rate scheduler.
        start_factor (float): Initial factor for the learning rate scheduler.
        total_iters (int): Total number of iterations for the learning rate scheduler.
        loss_func (nn.Module): The loss function used for training.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the ConvTasNet module.

        on_train_epoch_end() -> None:
            Hook method called at the end of each training epoch to log the training loss.

        on_validation_epoch_end() -> None:
            Hook method called at the end of each validation epoch to log the validation loss.

        _modify_grad_transfer_learning() -> None:
            Internal method to modify gradients during transfer learning.

        training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            Training step for the ConvTasNet module.

        validation_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            Validation step for the ConvTasNet module.

        configure_optimizers() -> Any:
            Method to configure the optimizer and optional learning rate scheduler.

    """

    def __init__(
            self,
            config: TrainingWaveformConfig,
            loss: nn.Module) -> None:
        """
        Forward pass of the ConvTasNet module.

        Args:
            x (torch.Tensor): Input waveform tensor of shape (batch_size, num_samples).

        Returns:
            torch.Tensor: Separated output tensor of shape (batch_size, num_sources, 1, num_samples).
        """
        super().__init__()
        model = CONVTASNET_BASE_LIBRI2MIX.get_model()
        self.models = nn.ModuleList(
            [model for _ in range(len(config.sources) // 2)])

        if config.transfer_learning:
            self._modify_grad_transfer_learning()

        self.train_loss = 0.0
        self.val_loss = 0.0

        self.lr = config.learning_rate
        self.wd = config.weight_decay
        self.eps = config.eps

        self.lr_scheduler = config.lr_scheduler
        self.start_factor = config.start_factor
        self.total_iters = config.total_iters

        self.loss_func = loss

    def on_train_epoch_end(self) -> None:
        """
        Hook method called at the end of each training epoch to log the training loss.
        """
        self.log(name='train_loss', value=self.train_loss)
        self.train_loss = 0.0

    def on_validation_epoch_end(self) -> None:
        """
        Hook method called at the end of each validation epoch to log the validation loss.
        """
        self.log("val_loss", self.val_loss)
        self.val_loss = 0.0

    def forward(self, x: torch.Tensor) -> Any:
        outs = [m(x) for m in self.models]
        outs = torch.cat(outs, dim=1)
        outs = outs.unsqueeze(2)
        return outs

    def _modify_grad_transfer_learning(self) -> None:
        """
        Internal method to modify gradients for transfer learning.
        """
        num_params = len([param for param in self.models[0].parameters()])
        # for transfer learning we will freeze all of the layers except two
        # last ones
        num_params -= 2
        for model in self.models:
            for param, _ in zip(model.parameters(), range(num_params)):
                param.requires_grad = False

    def training_step(self,
                      batch: Tuple[torch.Tensor,
                                   torch.Tensor],
                      batch_idx: int):
        """
        Training step for the ConvTasNet module.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch containing waveform and targets.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the training step.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_func(outputs, targets)
        self.log(name='train_step_loss', value=loss.detach(), on_step=True)
        self.train_loss += loss.item()
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor,
                                     torch.Tensor],
                        batch_idx: int):
        """
        Validation step for the ConvTasNet module.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch containing waveform and targets.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the validation step.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_func(outputs, targets)
        self.log(name='val_step_loss', value=loss.detach(), on_step=True)
        self.val_loss += loss.item()
        return loss

    def configure_optimizers(self) -> Any:
        """
        Method to configure the optimizer and optional learning rate scheduler.

        Returns:
            Any: Dictionary containing the optimizer and optionally the learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                self.parameters()),
            lr=self.lr,
            weight_decay=self.wd,
            eps=self.eps)
        optimizer_dict = {"optimizer": optimizer}
        if self.lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=self.start_factor,
                total_iters=self.total_iters)
            optimizer_dict["lr_scheduler"] = lr_scheduler

        return optimizer_dict
