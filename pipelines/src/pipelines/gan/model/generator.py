import logging

import torch
import torch.nn as nn

from pipelines.gan.model.layers import Conv1D, ResidualBlock, Transpose1dLayer

logger = logging.getLogger(__name__)

class CyclicWaveGanGenerator(nn.Module):
    """
    A generator model for Cyclic WaveGAN, designed for waveform generation.

    This model processes input waveforms through an encoder, transformation, and upsampling stages to generate output waveforms.
    It supports flexible slice lengths and optional batch normalization and upsampling configurations.

    Attributes:
        dim_mul (int): Dimensional multiplier for adjusting model sizes based on slice length.
        verbose (bool): Flag to enable detailed output of tensor shapes at different stages during forward propagation.
        encoder (nn.Sequential): Encoder network consisting of stacked Conv1D layers.
        transformation (nn.Sequential): Transformation network composed of ResidualBlock layers.
        upsampling (nn.Sequential): Upsampling network with stacked Transpose1dLayer layers.
    """
    def __init__(
            self,
            model_size: int = 64,
            num_channels: int = 1,
            shift_factor: int = 2,
            alpha: float = 0.2,
            verbose: bool = False,
            slice_len: int = 16384,
            use_batch_norm: bool = False,
            upsample: bool = True):
        """
        Initialize the CyclicWaveGanGenerator model.

        Args:
            model_size (int): Base model size to control the number of channels in intermediate layers.
            num_channels (int): Number of input and output channels (e.g., 1 for mono audio).
            shift_factor (int): Shift factor for cyclic convolution layers.
            alpha (float): Slope of the LeakyReLU activation function.
            verbose (bool): Whether to print the shape of intermediate tensors during the forward pass.
            slice_len (int): Length of input slices, must be one of [16384, 32768, 65536].
            use_batch_norm (bool): Whether to use batch normalization in the Conv1D and Transpose1dLayer layers.
            upsample (bool): Whether to enable upsampling in the Transpose1dLayer layers.
        """
        super(CyclicWaveGanGenerator, self).__init__()
        assert slice_len in [16384, 32768, 65536]  # used to predict longer utterances
        self.dim_mul = 16 if slice_len == 16384 else 32
        self.verbose = verbose
        encoder_conv = [
            Conv1D(
                num_channels,
                model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor),
            Conv1D(
                model_size,
                2 * model_size,
                25,
                stride=4,
                padding=13 if slice_len == 32768 else 11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor),
            # Conv1D(2 * model_size, 4*model_size , 25, stride=4, padding=13 if slice_len==32768 else 11, use_batch_norm=use_batch_norm, alpha=alpha,shift_factor=shift_factor)
        ]
        n_resblocks = 8
        if slice_len == 32768:
            encoder_conv.append(
                Conv1D(
                    2 * model_size,
                    (self.dim_mul * model_size) // 8,
                    25,
                    stride=2,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=shift_factor))
            n_resblocks = 9
        elif slice_len == 65536:
            encoder_conv.append(
                Conv1D(
                    2 * model_size,
                    (self.dim_mul * model_size) // 8,
                    25,
                    stride=4,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=shift_factor))
            n_resblocks = 9

        self.encoder = nn.Sequential(*encoder_conv)
        # encoder output 1 256 256

        transformation = []
        for _ in range(n_resblocks):
            transformation.append(ResidualBlock((self.dim_mul * model_size) // 8))
        self.transformation = nn.Sequential(*transformation)

        # Upsampling
        stride = 4
        if upsample:
            stride = 1
            upsample = 4

        deconv_layers = [
            Transpose1dLayer((self.dim_mul * model_size) // 8,
                             (self.dim_mul * model_size) // 16,
                             25,
                             stride,
                             upsample=upsample,
                             use_batch_norm=use_batch_norm),
            # Transpose1dLayer( (self.dim_mul* model_size) //8,  (self.dim_mul* model_size) //16, 25, stride, upsample=upsample,use_batch_norm=use_batch_norm),
        ]

        if slice_len == 16384:
            deconv_layers.append(
                Transpose1dLayer(
                    (self.dim_mul * model_size) // 16,
                    num_channels,
                    25,
                    stride,
                    upsample=upsample))
        elif slice_len == 32768:
            deconv_layers += [
                Transpose1dLayer(
                    (self.dim_mul * model_size) // 16,
                    model_size,
                    25,
                    stride,
                    upsample=upsample,
                    use_batch_norm=use_batch_norm),
                Transpose1dLayer(
                    model_size,
                    num_channels,
                    25,
                    2,
                    upsample=upsample)]
        elif slice_len == 65536:
            deconv_layers += [
                Transpose1dLayer(
                    (self.dim_mul * model_size) // 16,
                    model_size,
                    25,
                    stride,
                    upsample=upsample,
                    use_batch_norm=use_batch_norm),
                Transpose1dLayer(
                    model_size,
                    num_channels,
                    25,
                    stride,
                    upsample=upsample)]
        else:
            raise ValueError('slice_len {} value is not supported'.format(slice_len))
        self.upsampling = nn.Sequential(*deconv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the generator.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, slice_len).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_channels, slice_len).
        """
        x = self.encoder(x)
        if self.verbose:
            logger.log(x.shape)
        x = self.transformation(x)
        if self.verbose:
            logger.log(x.shape)
        x = self.upsampling(x)
        if self.verbose:
            logger.log(x.shape)
        return x
