from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from pipelines.types.training_spectrogram import TrainingSpectrogramConfig


class DoubleConvolutionBlock(pl.LightningModule):
    """
    DoubleConvolutionBlock: A PyTorch Lightning module defining a double convolution block.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.

    Attributes:
        conv1 (nn.Conv2d): First 2D convolution layer with ReLU activation.
        conv2 (nn.Conv2d): Second 2D convolution layer with ReLU activation.

    Methods:
        forward(inputs: torch.Tensor) -> torch.Tensor:
            Forward pass of the DoubleConvolutionBlock module.

    """

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1)

    def forward(self, inputs):
        x = nn.functional.relu(self.conv1(inputs))
        x = nn.functional.relu(self.conv2(x))
        return x


class EncoderBlock(pl.LightningModule):
    """
    EncoderBlock: A PyTorch Lightning module defining an encoder block.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.

    Attributes:
        conv (DoubleConvolutionBlock): DoubleConvolutionBlock used in the encoder.
        pool (nn.MaxPool2d): Max pooling layer.
        batch_norm (nn.BatchNorm2d): Batch normalization layer.

    Methods:
        forward(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Forward pass of the EncoderBlock module.

    """

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = DoubleConvolutionBlock(input_channels, output_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self, inputs):
        skip = self.conv(inputs)
        x = self.batch_norm(skip)
        x = self.pool(x)
        return skip, x


class DecoderBlock(pl.LightningModule):
    """
    DecoderBlock: A PyTorch Lightning module defining a decoder block.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.

    Attributes:
        conv (DoubleConvolutionBlock): DoubleConvolutionBlock used in the decoder.
        upconv (nn.ConvTranspose2d): Transposed 2D convolution layer (upsampling).
        batch_norm (nn.BatchNorm2d): Batch normalization layer.

    Methods:
        forward(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            Forward pass of the DecoderBlock module.

    """

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = DoubleConvolutionBlock(input_channels, output_channels)
        self.upconv = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self, inputs, skip):
        x = self.upconv(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        x = self.batch_norm(x)
        return x


class Unet(pl.LightningModule):
    """
    Unet: A PyTorch Lightning module implementing a U-Net for source separation.

    Args:
        config (TrainingSpectrogramConfig): The configuration object containing training settings.
        loss (nn.Module): The loss function used for training.

    Attributes:
        encoder1-4 (EncoderBlock): Encoder blocks for different stages of the U-Net.
        bottleneck (DoubleConvolutionBlock): Bottleneck block in the U-Net.
        decoder1-4 (DecoderBlock): Decoder blocks for different stages of the U-Net.
        classifier (nn.Conv2d): Final classifier layer.
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
        forward(inputs: torch.Tensor) -> torch.Tensor:
            Forward pass of the Unet module.

        training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            Training step for the Unet module.

        validation_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
            Validation step for the Unet module.

        on_train_epoch_end() -> None:
            Hook method called at the end of each training epoch to log the training loss.

        on_validation_epoch_end() -> None:
            Hook method called at the end of each validation epoch to log the validation loss.

        configure_optimizers() -> Any:
            Method to configure the optimizer and optional learning rate scheduler.

    """

    def __init__(self, config: TrainingSpectrogramConfig, loss: nn.Module):
        super().__init__()
        # Encoder
        self.encoder1 = EncoderBlock(config.input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        # Bottleneck
        self.bottleneck = DoubleConvolutionBlock(512, 1024)
        # Decoder
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        # Classifier
        self.classifier = nn.Conv2d(64, len(config.sources), kernel_size=1)

        self.train_loss = 0
        self.val_loss = 0

        self.lr = config.learning_rate
        self.wd = config.weight_decay
        self.eps = config.eps

        self.lr_scheduler = config.lr_scheduler
        self.start_factor = config.start_factor
        self.total_iters = config.total_iters

        # Loss
        self.loss_func = loss

    def forward(self, inputs):
        """
        Forward pass of the Unet module.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Separated output tensor of shape (batch_size, num_sources, height, width).
        """
        skip1, x = self.encoder1(inputs)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)
        x = self.bottleneck(x)
        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)
        outputs = self.classifier(x)
        outputs = outputs.unsqueeze(2)
        return outputs

    def training_step(self,
                      batch: Tuple[torch.Tensor,
                                   torch.Tensor],
                      batch_idx: int):
        """
        Training step for the Unet module.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch containing spectrogram and targets.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the training step.
        """

        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y_true)
        self.train_loss += loss.item()
        self.log("train_step_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the Unet module.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch containing spectrogram and targets.
            batch_idx (int): Batch index.
        """
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y_true)
        self.val_loss += loss.item()
        self.log('val_step_loss', loss, prog_bar=True)

    def on_train_epoch_end(self):
        """
        Hook method called at the end of each training epoch to log the training loss.
        """
        self.log("train_loss", self.train_loss)
        self.train_loss = 0

    def on_validation_epoch_end(self):
        """
        Hook method called at the end of each validation epoch to log the validation loss.
        """
        self.log("val_loss", self.val_loss)
        self.val_loss = 0

    def configure_optimizers(self):
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
