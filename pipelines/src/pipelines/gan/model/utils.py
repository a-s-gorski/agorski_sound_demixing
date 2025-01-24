from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pipelines.gan.config import TrainingGANConfig


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e., shifting the feature axis of a 3D tensor
    by a random integer in the range [-shift_factor, shift_factor], with
    reflection padding where necessary.

    Args:
        shift_factor (int): The maximum amount of shift along the feature axis.
    """

    def __init__(self, shift_factor: int):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for phase shuffling.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            Tensor: Tensor after applying phase shuffling.
        """
        if self.shift_factor == 0:
            return x

        # Generate random shifts for each sample in the batch
        k_list = torch.randint(
            -self.shift_factor, self.shift_factor + 1, (x.shape[0],), device=x.device
        ).tolist()

        # Group indices by their shift value to minimize shuffle operations
        k_map: Dict[int, List[int]] = {}
        for idx, k in enumerate(k_list):
            k_map.setdefault(k, []).append(idx)

        # Apply phase shuffling
        x_shuffle = x.clone()
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            elif k < 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, f"Shape mismatch: {x_shuffle.shape}, {x.shape}"
        return x_shuffle


def sample_noise(size: int, device: torch.device, config: TrainingGANConfig) -> Tensor:
    """
    Generates random noise for input to the GAN generator.

    Args:
        size (int): Number of samples in the batch.
        device (torch.device): The device to create the noise on.
        config (TrainingGANConfig): Configuration object containing the latent dimension.

    Returns:
        Tensor: Noise tensor of shape (size, config.noise_latent_dim).
    """
    z = torch.empty(size, config.noise_latent_dim, device=device).normal_()
    return z


def weights_init(m: nn.Module) -> None:
    """
    Initializes the weights of the model.

    Args:
        m (nn.Module): A module whose weights need to be initialized.
    """
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)


def update_optimizer_lr(
        optimizer: torch.optim.Optimizer,
        lr: float,
        decay: float) -> None:
    """
    Updates the learning rate of the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update.
        lr (float): The base learning rate.
        decay (float): The decay factor to apply to the learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * decay


def gradients_status(model: nn.Module, flag: bool) -> None:
    """
    Toggles the requires_grad status of all parameters in a model.

    Args:
        model (nn.Module): The model whose parameters will be updated.
        flag (bool): If True, gradients are enabled; if False, gradients are disabled.
    """
    for p in model.parameters():
        p.requires_grad = flag
