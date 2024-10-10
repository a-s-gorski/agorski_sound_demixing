from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class GenerativeModel(Enum):
    """Enum representing the types of generative models.

    Attributes:
        VAE (str): Value representing the Variational Autoencoder (VAE) generative model.
        GAN (str): Value representing the Generative Adversarial Network (GAN) generative model.
    """
    VAE = 'vae'
    GAN = 'gan'


class TrainingGenerativeConfig(BaseModel):
    """Configuration for training generative models.

    Attributes:
        model (GenerativeModel): The type of generative model to train (VAE or GAN).
        sources (List[str]): A list of strings representing the names of the audio sources.
        input_channels (int): The number of input channels in the audio signals.
        output_channels (int): The number of output channels in the generated audio signals.
        batch_size (int): The batch size to use during training.
        num_workers (int): The number of workers for data loading during training.
        latent_dim (int): The dimension of the latent space for VAE models.
        layer_n (int): The number of layers in the generative model.
        hidden (int): The number of hidden units in each layer of the generative model.
        signal_len (int): The length of the input audio signal.
        learning_rate (float): The learning rate for the optimizer during training.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer during training.
        eps (float): The epsilon value for numerical stability in VAE models.
        max_epochs (int): The maximum number of epochs for training.
        lr_scheduler (bool): A flag indicating whether to use a learning rate scheduler.
        start_factor (float): The factor by which the learning rate is multiplied at the beginning of training.
        total_iters (int): The total number of iterations for training (used for the lr_scheduler).
        early_stopping (bool): A flag indicating whether to use early stopping during training.
        patience (int): The number of epochs to wait before early stopping if the validation loss does not improve.
        quality (str): The quality metric to optimize during training (e.g., 'mse', 'spectral_loss', etc.).
        save_checkpoint (bool): A flag indicating whether to save checkpoints during training.
        checkpoint_output_path (str): The path to save the checkpoints during training.
        save_top_k (int): The number of best checkpoints to keep based on the validation loss.
        every_n_epochs (int): The frequency of saving checkpoints (in epochs).
        log_mlflow (bool): A flag indicating whether to log training metrics using mlflow.
    """
    model: GenerativeModel
    sources: List[str]
    input_channels: int
    output_channels: int
    batch_size: Optional[int] = 4
    num_workers: Optional[int] = 12
    latent_dim: Optional[int] = 3
    layer_n: Optional[int] = 3
    hidden: Optional[int] = 64
    signal_len: Optional[int] = 10000
    learning_rate: Optional[float] = 0.0001
    weight_decay: Optional[float] = 0.0
    eps: Optional[float] = 1e-8
    max_epochs: Optional[int] = 20
    lr_scheduler: Optional[bool] = False
    start_factor: Optional[float] = 0.01
    total_iters: Optional[int] = 3
    early_stopping: Optional[bool] = False
    patience: Optional[int] = 3
    quality: Optional[str] = "val_loss"
    save_checkpoint: Optional[bool] = False
    checkpoint_output_path: Optional[str] = "."
    save_top_k: Optional[int] = 1
    every_n_epochs: Optional[int] = 3
    log_mlflow: Optional[bool] = False
