import torch
import torch.nn as nn

from pipelines.gan.model.layers import Conv1D


class WaveGANDiscriminator(nn.Module):
    """
    A discriminator model for WaveGAN, designed for processing 1D audio signals.

    This discriminator uses a series of convolutional layers followed by a fully connected layer
    to classify audio slices as real or fake. It supports variable slice lengths and optional batch
    normalization.

    Attributes:
        model_size (int): Base number of filters in the convolutional layers. Default is 64.
        ngpus (int): Number of GPUs to use. Default is 1.
        num_channels (int): Number of input channels. Default is 1.
        shift_factor (int): Shift factor for temporal convolutions. Default is 2.
        alpha (float): Leaky ReLU slope. Default is 0.2.
        verbose (bool): If True, prints the shape of tensors at each layer during the forward pass.
        slice_len (int): Length of the input slices. Must be one of [16384, 32768, 65536].
        use_batch_norm (bool): Whether to use batch normalization in the convolutional layers.
        fc_input_size (int): Size of the input to the fully connected layer. Determined by the slice length.
        conv_layers (nn.ModuleList): List of convolutional layers.
        fc1 (nn.Linear): Final fully connected layer producing a single scalar output.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the discriminator.
    """

    def __init__(self, model_size=64, ngpus=1, num_channels=1, shift_factor=2,
                 alpha=0.2, verbose=False, slice_len=16384, use_batch_norm=False):
        super(WaveGANDiscriminator, self).__init__()
        assert slice_len in [16384, 32768, 65536]  # used to predict longer utterances

        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.use_batch_norm = use_batch_norm
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        conv_layers = [
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
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor),
            Conv1D(
                2 * model_size,
                4 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor),
            Conv1D(
                4 * model_size,
                8 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor),
            Conv1D(
                8 * model_size,
                16 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=0 if slice_len == 16384 else shift_factor)]
        self.fc_input_size = 256 * model_size
        if slice_len == 32768:
            conv_layers.append(
                Conv1D(
                    16 * model_size,
                    32 * model_size,
                    25,
                    stride=2,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=0))
            self.fc_input_size = 480 * model_size
        elif slice_len == 65536:
            conv_layers.append(
                Conv1D(
                    16 * model_size,
                    32 * model_size,
                    25,
                    stride=4,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=0))
            self.fc_input_size = 512 * model_size

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x) -> torch.Tensor:
        """
        Performs a forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, slice_len).

        Returns:
            torch.Tensor: A scalar output tensor of shape (batch_size, 1), representing the discriminator's score for each input slice.
        """
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(x.shape)
        x = x.view(-1, self.fc_input_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)
