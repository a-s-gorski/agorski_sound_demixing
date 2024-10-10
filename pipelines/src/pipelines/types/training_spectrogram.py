from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class SpectrogramModel(Enum):
    """
    Enum class representing the types of spectrogram models.

    Attributes:
        UNET (str): U-Net model for spectrogram processing.
    """
    UNET = 'unet'


class SpectrogramLoss(Enum):
    """
    Enum class representing the types of spectrogram loss functions.

    Attributes:
        MSE (str): Mean Squared Error loss function.
        MAE (str): Mean Absolute Error loss function.
        SDR (str): Signal-to-Distortion Ratio loss function.
    """
    MSE = 'mse'
    MAE = 'mae'
    SDR = 'sdr'


class TrainingSpectrogramConfig(BaseModel):
    """
    A configuration model for training spectrogram models.

    Attributes:
        model (SpectrogramModel): The type of spectrogram model being used.
        loss (SpectrogramLoss): The loss function used during training.
        batch_size (Optional[int]): The batch size for training. Defaults to 4.
        num_workers (Optional[int]): The number of CPU workers used for data loading. Defaults to 1.
        learning_rate (Optional[float]): The learning rate for the optimizer. Defaults to 0.001.
        weight_decay (Optional[float]): The weight decay for the optimizer. Defaults to 0.0.
        eps (Optional[float]): The epsilon value used for numerical stability. Defaults to 1e-8.
        max_epochs (Optional[int]): The maximum number of training epochs. Defaults to 20.
        lr_scheduler (Optional[bool]): Whether to use a learning rate scheduler or not. Defaults to False.
        start_factor (Optional[float]): The factor to start annealing the learning rate. Defaults to 0.01.
        total_iters (Optional[int]): The total number of iterations for the learning rate scheduler. Defaults to 5.
        early_stopping (Optional[bool]): Whether to use early stopping during training or not. Defaults to False.
        patience (Optional[int]): The number of epochs with no improvement after which training will be stopped.
                                 Defaults to 3.
        quality (Optional[str]): The metric to track for early stopping. Defaults to "val_loss".
        save_checkpoint (Optional[bool]): Whether to save checkpoints during training or not. Defaults to False.
        checkpoint_output_path (Optional[str]): The output path for saving checkpoints. Defaults to ".".
        save_top_k (Optional[int]): The number of best models to keep when saving checkpoints. Defaults to 1.
        every_n_epochs (Optional[int]): Save a checkpoint every N epochs. Defaults to 3.
        log_mlflow (Optional[bool]): Whether to log training metrics using MLflow or not. Defaults to False.
    """
    model: SpectrogramModel
    loss: SpectrogramLoss
    input_channels: int
    sources: List[str]
    batch_size: Optional[int] = 4
    num_workers: Optional[int] = 1
    learning_rate: Optional[float] = 0.001
    weight_decay: Optional[float] = 0.0
    eps: Optional[float] = 1e-8
    max_epochs: Optional[int] = 20
    lr_scheduler: Optional[bool] = False
    start_factor: Optional[float] = 0.01
    total_iters: Optional[int] = 5
    early_stopping: Optional[bool] = False
    patience: Optional[int] = 3
    quality: Optional[str] = "val_loss"
    save_checkpoint: Optional[bool] = False
    checkpoint_output_path: Optional[str] = "."
    save_top_k: Optional[int] = 1
    every_n_epochs: Optional[int] = 3 
    log_mlflow: Optional[bool] = False
