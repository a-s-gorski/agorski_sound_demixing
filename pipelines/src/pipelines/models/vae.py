import pytorch_lightning as pl
import torch
from torch import nn

from pipelines.types.training_signal_enriched import TrainingGenerativeConfig


class Encoder(nn.Module):
    """Encoder module for the Variational Autoencoder (VAE).

    Args:
        num_input_channels (int): Number of input channels.
        hidden (int): Number of hidden channels in the encoder.

    Attributes:
        net (nn.Sequential): Sequential model containing convolutional layers and GELU activation.
    """

    def __init__(self, num_input_channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                num_input_channels,
                2 * hidden,
                kernel_size=3,
                padding=1,
                stride=2),
            nn.GELU(),
            nn.Conv1d(
                2 * hidden,
                2 * hidden,
                kernel_size=3,
                padding=1),
            nn.GELU(),
            nn.Conv1d(
                2 * hidden,
                hidden,
                kernel_size=3,
                padding=1,
                stride=2),
            nn.GELU(),
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1),
            nn.GELU(),
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Decoder module for the Variational Autoencoder (VAE).

    Args:
        num_output_channels (int): Number of output channels.
        hidden (int): Number of hidden channels in the decoder.
        latent_dim (int): Dimension of the latent space.
        signal_len (int): Length of the input signal.

    Attributes:
        net (nn.Sequential): Sequential model containing transposed convolutional layers and GELU activation.
    """

    def __init__(
            self,
            num_output_channels: int,
            hidden: int,
            latent_dim: int,
            signal_len: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                latent_dim,
                signal_len // 8),
            nn.ConvTranspose1d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(
                hidden,
                2 * hidden,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(
                2 * hidden,
                2 * hidden,
                padding=1,
                kernel_size=3,
                stride=2,
                output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(
                2 * hidden,
                num_output_channels,
                padding=1,
                kernel_size=3))

    def forward(self, x):
        return self.net(x)


class VAE(pl.LightningModule):
    """Variational Autoencoder (VAE) model.

    Args:
        config (TrainingGenerativeConfig): Configuration for the VAE.

    Attributes:
        enc (Encoder): The encoder module.
        dec (Decoder): The decoder module.
        mean (nn.Linear): Linear layer for the mean of the latent space.
        var (nn.Linear): Linear layer for the variance of the latent space.
        recon_loss_func (nn.MSELoss): Mean Squared Error loss function for reconstruction.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for the optimizer.
        eps (float): Epsilon value for numerical stability.
        lr_scheduler (bool): Whether to use a linear learning rate scheduler.
        start_factor (float): Start factor for the linear learning rate scheduler.
        total_iters (int): Total number of iterations for the linear learning rate scheduler.
        train_elbo (float): Cumulative training Evidence Lower Bound (ELBO) loss.
        train_kl (float): Cumulative training Kullback-Leibler (KL) divergence loss.
        train_recon (float): Cumulative training reconstruction loss.
        val_elbo (float): Cumulative validation Evidence Lower Bound (ELBO) loss.
        val_kl (float): Cumulative validation Kullback-Leibler (KL) divergence loss.
        val_recon (float): Cumulative validation reconstruction loss.
    """

    def __init__(self, config: TrainingGenerativeConfig) -> None:
        # def __init__(self, num_input_channels: int, num_output_channels: int,
        # hidden: int, latent_dim: int, signal_len: int) -> None:
        super().__init__()
        self.enc = Encoder(config.input_channels, config.hidden)
        self.dec = Decoder(
            config.output_channels,
            config.hidden,
            config.latent_dim,
            config.signal_len)
        self.mean = nn.Linear(config.signal_len // 8, config.latent_dim)
        self.var = nn.Linear(config.signal_len // 8, config.latent_dim)
        self.recon_loss_func = nn.MSELoss()

        self.lr = config.learning_rate
        self.wd = config.weight_decay
        self.eps = config.eps

        self.lr_scheduler = config.lr_scheduler
        self.start_factor = config.start_factor
        self.total_iters = config.total_iters

        self.train_elbo = 0.0
        self.train_kl = 0.0
        self.train_recon = 0.0

        self.val_elbo = 0.0
        self.val_kl = 0.0
        self.val_recon = 0.0

    def kl_divergence(
            self,
            z: torch.Tensor,
            mean: torch.Tensor,
            var: torch.Tensor):
        """Compute the Kullback-Leibler (KL) divergence between the approximate posterior and the prior.

        Args:
            z (torch.Tensor): Sampled latent variables from the approximate posterior.
            mean (torch.Tensor): Mean of the latent space from the encoder.
            var (torch.Tensor): Variance of the latent space from the encoder.

        Returns:
            torch.Tensor: The computed KL divergence.
        """
        kl = 0.5 * torch.sum(1 + torch.log(var ** 2) - mean ** 2 - var ** 2)
        return kl

    def forward(self, x):
        """Forward pass of the Variational Autoencoder (VAE).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the mean, variance,
            sampled latent variables, and reconstructed output from the decoder.
        """
        x = self.enc(x)
        mean = self.mean(x)
        var = self.var(x)
        # reparametrize
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        x_hat = self.dec(z)
        return mean, var, z, x_hat

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Training step for the Variational Autoencoder (VAE).

        Args:
            batch (torch.Tensor): Input batch of data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed Evidence Lower Bound (ELBO) loss.
        """
        x, x_target = batch
        mean, var, z, x_hat = self.forward(x)
        recon_loss = self.recon_loss_func(x_target, x_hat)
        kl = self.kl_divergence(z, mean, var)
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        self.log(name='train_step_loss', value=elbo, prog_bar=True)
        self.log(name='train_step_recon_loss', value=recon_loss, prog_bar=True)
        self.log(name='train_step_kl_loss', value=kl, prog_bar=True)

        self.train_elbo += elbo.item()
        self.train_recon += recon_loss.item()
        self.train_kl += kl.item()

        return elbo

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Validation step for the Variational Autoencoder (VAE).

        Args:
            batch (torch.Tensor): Input batch of data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed Evidence Lower Bound (ELBO) loss.
        """
        x, x_target = batch
        mean, var, z, x_hat = self.forward(x)
        recon_loss = self.recon_loss_func(x_target, x_hat)
        kl = self.kl_divergence(z, mean, var)
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        self.log(name='val_step_loss', value=elbo, prog_bar=True)
        self.log(name='val_step_recon_loss', value=recon_loss, prog_bar=True)
        self.log(name='val_step_kl_loss', value=kl, prog_bar=True)

        self.val_elbo += elbo.item()
        self.val_recon += recon_loss.item()
        self.val_kl += kl.item()

        return elbo

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch to log and reset the cumulative training losses."""
        self.log(name='train_epoch_loss', value=self.train_elbo)
        self.log(name='train_epoch_recon_loss', value=self.train_recon)
        self.log(name='train_epoch_kl_loss', value=self.train_kl)
        self.train_elbo, self.train_recon, self.train_kl = 0.0, 0.0, 0.0

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch to log and reset the cumulative validation losses."""
        self.log(name='val_epoch_loss', value=self.val_elbo)
        self.log(name='val_epoch_recon_loss', value=self.val_recon)
        self.log(name='val_epoch_kl_loss', value=self.val_kl)
        self.val_elbo, self.val_recon, self.val_kl = 0.0, 0.0, 0.0

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler for training.

        Returns:
            Dict[str, Any]: Dictionary containing the optimizer and optionally the learning rate scheduler.
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
