from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from pipelines.types.training_signal_enriched import TrainingGenerativeConfig


class ConvBlock(pl.LightningModule):
    """Convolutional Block Module.

    Args:
        in_layer (int): Number of input channels.
        out_layer (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride value for the convolutional operation.
        dilation (int): Dilation value for the convolutional operation.

    Attributes:
        conv1 (nn.Conv1d): 1D Convolutional layer.
        bn (nn.BatchNorm1d): Batch Normalization layer.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_layer,
            out_layer,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=3,
            bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class SeBlock(pl.LightningModule):
    """SE Block Module.

    Args:
        in_layer (int): Number of input channels.
        out_layer (int): Number of output channels.

    Attributes:
        conv1 (nn.Conv1d): 1D Convolutional layer.
        conv2 (nn.Conv1d): 1D Convolutional layer.
        fc (nn.Linear): Linear layer for global pooling.
        fc2 (nn.Linear): Linear layer for channel-wise scaling.
        relu (nn.ReLU): ReLU activation function.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self, in_layer, out_layer):
        super(SeBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_layer,
            out_layer // 8,
            kernel_size=1,
            padding=0)
        self.conv2 = nn.Conv1d(
            out_layer // 8,
            in_layer,
            kernel_size=1,
            padding=0)
        self.fc = nn.Linear(1, out_layer // 8)
        self.fc2 = nn.Linear(out_layer // 8, out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_se = nn.functional.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)

        x_out = torch.add(x, x_se)
        return x_out


class ReBlock(pl.LightningModule):
    """Residual Block Module.

    Args:
        in_layer (int): Number of input channels.
        out_layer (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation value for the convolutional operation.

    Attributes:
        cbr1 (ConvBlock): First Convolutional Block.
        cbr2 (ConvBlock): Second Convolutional Block.
        seblock (SeBlock): Squeeze-and-Excitation Block.
    """

    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(ReBlock, self).__init__()

        self.cbr1 = ConvBlock(in_layer, out_layer, kernel_size, 1, dilation)
        self.cbr2 = ConvBlock(out_layer, out_layer, kernel_size, 1, dilation)
        self.seblock = SeBlock(out_layer, out_layer)

    def forward(self, x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out


class Generator(pl.LightningModule):
    """Generator Model for UnetGAN.

    Args:
        input_dim (int): Number of input channels.
        output_dim (int): Number of output channels.
        layer_n (int): Number of initial channels for the layers.
        kernel_size (int): Size of the convolutional kernel.
        depth (int): Depth of the Residual Blocks.

    Attributes:
        AvgPool1D1 (nn.AvgPool1d): First Average Pooling layer.
        AvgPool1D2 (nn.AvgPool1d): Second Average Pooling layer.
        AvgPool1D3 (nn.AvgPool1d): Third Average Pooling layer.
        layer1 (nn.Sequential): First Down-Sampling Layer.
        layer2 (nn.Sequential): Second Down-Sampling Layer.
        layer3 (nn.Sequential): Third Down-Sampling Layer.
        layer4 (nn.Sequential): Fourth Down-Sampling Layer.
        layer5 (nn.Sequential): Fifth Down-Sampling Layer.
        cbr_up1 (ConvBlock): First Up-Sampling Convolutional Block.
        cbr_up2 (ConvBlock): Second Up-Sampling Convolutional Block.
        cbr_up3 (ConvBlock): Third Up-Sampling Convolutional Block.
        upsample (nn.Upsample): Up-sampling layer using nearest neighbor interpolation.
        upsample1 (nn.Upsample): Additional up-sampling layer using nearest neighbor interpolation.
        outcov (nn.Conv1d): Output Convolutional layer.
    """

    def __init__(self, input_dim, output_dim, layer_n, kernel_size, depth):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_dim = output_dim

        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)

        self.layer1 = self.down_layer(
            self.input_dim, self.layer_n, self.kernel_size, 1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(
            self.layer_n * 2), self.kernel_size, 5, 2)
        self.layer3 = self.down_layer(int(
            self.layer_n * 2) + int(self.input_dim), int(self.layer_n * 3), self.kernel_size, 5, 2)
        self.layer4 = self.down_layer(int(
            self.layer_n * 3) + int(self.input_dim), int(self.layer_n * 4), self.kernel_size, 5, 2)
        self.layer5 = self.down_layer(int(
            self.layer_n * 4) + int(self.input_dim), int(self.layer_n * 5), self.kernel_size, 4, 2)

        self.cbr_up1 = ConvBlock(
            int(self.layer_n * 7), int(self.layer_n * 3), self.kernel_size, 1, 1)
        self.cbr_up2 = ConvBlock(
            int(self.layer_n * 5), int(self.layer_n * 2), self.kernel_size, 1, 1)
        self.cbr_up3 = ConvBlock(
            int(self.layer_n * 3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')

        self.outcov = nn.Conv1d(
            self.layer_n,
            output_dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=3)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(ConvBlock(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(ReBlock(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x):

        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x)

        x = torch.cat([out_2, pool_x2], 1)
        x = self.layer4(x)


        up = self.upsample1(x)
        up = torch.cat([up, out_2], 1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up, out_1], 1)
        up = self.cbr_up2(up)

        up = self.upsample(up)
        up = torch.cat([up, out_0], 1)
        up = self.cbr_up3(up)

        out = self.outcov(up)

        return out


class Discriminator(pl.LightningModule):
    """Discriminator Model for UnetGAN.

    Args:
        signal_len (int): Length of the input signal.

    Attributes:
        model (nn.Sequential): Sequential model containing linear layers and activation functions.
    """

    def __init__(self, signal_len: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(signal_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class UnetGAN(pl.LightningModule):
    """UnetGAN Model for Signal Generation.

    Args:
        config (TrainingGenerativeConfig): Configuration for the generator.
        device (torch.device): The device where the model is run.

    Attributes:
        gen (Generator): The Generator model.
        disc (Discriminator): The Discriminator model.
        loss_func (nn.MSELoss): Mean Squared Error loss function.
        adv_loss_func (nn.BCELoss): Binary Cross Entropy loss function for adversarial loss.
        lr (float): Learning rate for the optimizers.
        train_generator_loss (float): Cumulative training generator loss.
        train_discriminator_loss (float): Cumulative training discriminator loss.
        train_reconstruction_loss (float): Cumulative training reconstruction loss.
        val_generator_loss (float): Cumulative validation generator loss.
        val_discriminator_loss (float): Cumulative validation discriminator loss.
        val_reconstruction_loss (float): Cumulative validation reconstruction loss.
    """

    def __init__(
            self,
            config: TrainingGenerativeConfig,
            device: torch.device) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.gen = Generator(
            config.input_channels,
            config.output_channels,
            config.layer_n,
            7,
            3)
        self.disc = Discriminator(config.signal_len)
        self.loss_func = nn.MSELoss()
        self.adv_loss_func = nn.BCELoss()
        self.lr = config.learning_rate

        self.train_generator_loss = 0.0
        self.train_discriminator_loss = 0.0
        self.train_reconstruction_loss = 0.0
        self.val_generator_loss = 0.0
        self.val_discriminator_loss = 0.0
        self.val_reconstruction_loss = 0.0

        self.device_ = device

    def generator_loss(self,
                       real: torch.Tensor,
                       targets: torch.Tensor,
                       fake: torch.Tensor) -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        reconstruction_loss = self.loss_func(fake, targets)
        real_logits = self.disc(real)
        real_loss = self.adv_loss_func(
            real_logits, torch.zeros(
                size=real_logits.shape).to(
                self.device_))
        fake_logits = self.disc(fake)
        fake_loss = self.adv_loss_func(
            fake_logits, torch.ones(
                size=fake_logits.shape).to(
                self.device_))
        return reconstruction_loss + real_loss + fake_loss, reconstruction_loss

    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor):
        real_logits = self.disc(real)
        real_loss = self.adv_loss_func(
            real_logits, torch.ones(
                size=real_logits.shape).to(
                self.device_))
        fake_logits = self.disc(fake)
        fake_loss = self.adv_loss_func(
            fake_logits, torch.zeros(
                size=fake_logits.shape).to(
                self.device_))
        return real_loss + fake_loss

    def configure_optimizers(self):
        lr = self.lr
        b1 = 0.0001
        b2 = 0.0001

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def forward(self, x):
        return self.gen(x)

    def training_step(self,
                      batch: Tuple[torch.Tensor,
                                   torch.Tensor],
                      batch_idx) -> None:
        real, targets = batch
        fake = self(real)
        optimizer_g, optimizer_d = self.optimizers()
        g_loss, recon_loss = self.generator_loss(real, targets, fake)

        self.log("train_step_reconstruction_loss", recon_loss)
        self.train_reconstruction_loss += recon_loss
        self.log("train_step_generator_loss", g_loss, prog_bar=True)
        self.train_generator_loss += g_loss
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        fake = self(real)
        self.toggle_optimizer(optimizer_d)
        d_loss = self.discriminator_loss(real, fake)
        self.log("train_step_discriminator_loss", d_loss, prog_bar=True)
        self.train_discriminator_loss += d_loss
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self,
                        batch: Tuple[torch.Tensor,
                                     torch.Tensor],
                        batch_idx: int) -> None:
        real, targets = batch
        fake = self(real)
        g_loss, recon_loss = self.generator_loss(real, targets, fake)
        self.log("val_step_reconstruction_loss", recon_loss)
        self.val_reconstruction_loss += recon_loss
        self.log("val_step_generator_loss", g_loss)
        self.val_generator_loss += g_loss
        fake = self(real)
        d_loss = self.discriminator_loss(real, fake)
        self.log("val_step_discriminator_loss", d_loss)
        self.val_discriminator_loss += d_loss

    def on_train_epoch_end(self) -> None:
        self.log('train_recon_loss', self.train_reconstruction_loss)
        self.log('train_generator_loss', self.train_generator_loss)
        self.log('train_discriminator_loss', self.train_discriminator_loss)
        self.train_reconstruction_loss = 0.0
        self.train_generator_loss = 0.0
        self.train_discriminator_loss = 0.0

    def on_validation_epoch_end(self) -> None:
        self.log('val_recon_loss', self.val_reconstruction_loss)
        self.log('val_generator_loss', self.val_generator_loss)
        self.log('val_discriminator_loss', self.val_discriminator_loss)
        self.val_reconstruction_loss = 0.0
        self.val_generator_loss = 0.0
        self.val_discriminator_loss = 0.0
