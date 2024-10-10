from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from pipelines.types.training_waveform import TrainingWaveformConfig


class BiLSTM(pl.LightningModule):
    """
    BiLSTM: A PyTorch Lightning module implementing a Bidirectional LSTM for source seperation.

    Args:
        config (TrainingWaveformConfig): The configuration object containing training settings.
        loss (nn.Module): The loss function used for training.

    Attributes:
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of LSTM layers in the model.
        lstms (nn.ModuleList): A list of Bidirectional LSTM layers for each source.
        fcs (nn.ModuleList): A list of fully connected layers for each source.
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
            Forward pass of the BiLSTM module.

        training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            Training step for the BiLSTM module.

        validation_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            Validation step for the BiLSTM module.

        on_train_epoch_end() -> None:
            Hook method called at the end of each training epoch to log the training loss.

        on_validation_epoch_end() -> None:
            Hook method called at the end of each validation epoch to log the validation loss.

        configure_optimizers() -> Any:
            Method to configure the optimizer and optional learning rate scheduler.

    """

    def __init__(
            self,
            config: TrainingWaveformConfig,
            loss: nn.Module) -> None:
        """
        Initialize the BiLSTM module.

        Args:
            config (TrainingWaveformConfig): The configuration object containing training settings.
            loss (nn.Module): The loss function used for training.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstms = nn.ModuleList([nn.LSTM(input_size=config.num_channels,
                                            hidden_size=config.hidden_size,
                                            num_layers=config.num_layers,
                                            batch_first=True,
                                            bidirectional=config.bidirectional,
                                            dropout=config.dropout) for _ in range(len(config.sources))])
        self.fcs = nn.ModuleList(
            [
                nn.Linear(
                    config.hidden_size *
                    2 if config.bidirectional else config.hidden_size,
                    out_features=config.num_channels) for _ in range(
                    len(
                        config.sources))])

        self.train_loss = 0.0
        self.val_loss = 0.0

        self.lr = config.learning_rate
        self.wd = config.weight_decay
        self.eps = config.eps

        self.lr_scheduler = config.lr_scheduler
        self.start_factor = config.start_factor
        self.total_iters = config.total_iters

        self.loss_func = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BiLSTM module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_samples).

        Returns:
            torch.Tensor: Separated output tensor of shape (batch_size, num_sources, num_samples).
        """
        x = x.reshape(-1, x.shape[-1], x.shape[-2])
        outs = [lstm(x) for lstm in self.lstms]
        outs = [fc(out[0]) for fc, out in zip(self.fcs, outs)]
        outs = [out.reshape(-1, out.shape[-1], out.shape[-2]
                            ).unsqueeze(1) for out in outs]
        outs = torch.cat(outs, dim=1)
        return outs

    def training_step(self,
                      batch: Tuple[torch.Tensor,
                                   torch.Tensor],
                      batch_idx):
        """
        Training step for the BiLSTM module.

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

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the BiLSTM module.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch containing waveform and targets.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the validation step.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_func(outputs, targets)
        self.log(
            name='val_step_loss',
            value=loss.detach(),
            on_step=True,
            prog_bar=True)
        self.val_loss += loss.item()
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Hook method called at the end of each training epoch to log the training loss.
        """
        self.log(
            name='train_loss',
            value=self.train_loss,
            on_epoch=True,
            prog_bar=True)
        self.train_loss = 0.0

    def on_validation_epoch_end(self) -> None:
        """
        Hook method called at the end of each validation epoch to log the validation loss.
        """
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.val_loss = 0.0

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
