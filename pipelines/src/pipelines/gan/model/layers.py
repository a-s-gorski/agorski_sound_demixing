import torch.nn as nn
import torch.nn.functional as F

from pipelines.gan.model.utils import PhaseShuffle


class Conv1D(nn.Module):
    """
    A 1D convolutional layer with optional Batch Normalization, Leaky ReLU activation,
    Phase Shuffling, and Dropout.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        alpha (float, optional): Negative slope for the Leaky ReLU activation. Default: 0.2.
        shift_factor (int, optional): Shift factor for Phase Shuffle. Default: 2.
        stride (int, optional): Stride for the convolution. Default: 4.
        padding (int, optional): Padding for the convolution. Default: 11.
        use_batch_norm (bool, optional): Whether to use Batch Normalization. Default: False.
        drop_prob (float, optional): Dropout probability. Default: 0.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: int,
            alpha: float = 0.2,
            shift_factor: int = 2,
            stride: int = 4,
            padding: int = 11,
            use_batch_norm: bool = False,
            drop_prob: int = 0):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=padding)
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.phase_shuffle = PhaseShuffle(shift_factor)
        self.alpha = alpha
        self.use_batch_norm = use_batch_norm
        self.use_phase_shuffle = shift_factor == 0
        self.use_drop = drop_prob > 0
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        """
        Forward pass of the Conv1D module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        if self.use_phase_shuffle:
            x = self.phase_shuffle(x)
        if self.use_drop:
            x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block composed of two Conv1D layers.

    Args:
        in_features (int): Number of input (and output) features.
        use_batch_norm (bool, optional): Whether to use Batch Normalization. Default: True.
        alpha (float, optional): Negative slope for the Leaky ReLU activation. Default: 0.2.
        shift_factor (int, optional): Shift factor for Phase Shuffle. Default: 2.
    """

    def __init__(
            self,
            in_features: int,
            use_batch_norm: bool = True,
            alpha: float = 0.2,
            shift_factor: int = 2):
        super(ResidualBlock, self).__init__()
        conv_blocks = [
            Conv1D(
                in_features,
                in_features,
                21,
                stride=1,
                padding=10,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor),
            Conv1D(
                in_features,
                in_features,
                21,
                stride=1,
                padding=10,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor)]
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        """
        Forward pass of the Transpose1dLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor.
        """
        output = x
        for conv in self.conv_blocks:
            output = conv(output)
        return x + output


class Transpose1dLayer(nn.Module):
    """
    A 1D transpose convolutional layer with optional upsampling and Batch Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the transpose convolution.
        padding (int, optional): Padding for the transpose convolution. Default: 11.
        upsample (float, optional): Upsampling factor. If provided, nearest-neighbor upsampling
                                    is applied before convolution. Default: None.
        output_padding (int, optional): Additional padding for the transpose convolution. Default: 1.
        use_batch_norm (bool, optional): Whether to use Batch Normalization. Default: False.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int = 11,
            upsample: bool = None,
            output_padding=1,
            use_batch_norm: bool = False):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding)
        batch_norm = nn.BatchNorm1d(out_channels)
        if self.upsample:
            operation_list = [
                reflection_pad,
                conv1d
            ]
        else:
            operation_list = [
                Conv1dTrans
            ]

        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        """
        Forward pass of the Transpose1dLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.upsample:
            # recommended by wavgan paper to use nearest upsampling
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')
        return self.transpose_ops(x)
