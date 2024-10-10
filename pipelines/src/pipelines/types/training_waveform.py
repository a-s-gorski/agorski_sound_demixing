from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class WaveformModel(Enum):
    """
    Enum class representing the types of waveform models.

    Attributes:
        LSTM (str): Long Short-Term Memory (LSTM) model for waveform processing.
        HDEMUCS (str): HDE-MuCS model for waveform processing.
        CONVTASNET (str): Conv-TasNet model for waveform processing.
    """
    LSTM = "lstm"
    HDEMUCS = "hdemucs"
    CONVTASNET = "convtasnet"


class WaveformLoss(Enum):
    """
    Enum class representing the types of waveform loss functions.

    Attributes:
        MSE (str): Mean Squared Error loss function.
        MAE (str): Mean Absolute Error loss function.
        SDR (str): Signal-to-Distortion Ratio loss function.
    """
    MSE = "mse"
    MAE = "mae"
    SDR = "sdr"


class TrainingWaveformConfig(BaseModel):
    """
    A configuration model for training waveform models.

    Attributes:
        model (WaveformModel): The type of waveform model being used.
        loss (WaveformLoss): The loss function used during training.
        sources (List[str]): List of sources used in training.
        batch_size (Optional[int]): The batch size for training. Default is 4.
        num_layers (Optional[int]): The number of layers in the model. Default is 3.
        hidden_size (Optional[int]): The size of the hidden state in the model. Default is 64.
        num_workers (Optional[int]): The number of CPU workers used for data loading. Default is 1.
        bidirectional (Optional[bool]): Whether the model uses bidirectional layers or not. Default is True.
        transfer_learning (Optional[bool]): Whether to use transfer learning during training or not. Default is True.
        dropout (Optional[float]): The dropout probability used in the model. Default is 0.0.
        learning_rate (Optional[float]): The learning rate for the optimizer. Default is 0.001.
        weight_decay (Optional[float]): The weight decay for the optimizer. Default is 0.0.
        eps (Optional[float]): The epsilon value used for numerical stability. Default is 1e-8.
        max_epochs (Optional[int]): The maximum number of training epochs. Default is 20.
        lr_scheduler (Optional[bool]): Whether to use a learning rate scheduler or not. Default is False.
        start_factor (Optional[float]): The factor to start annealing the learning rate. Default is 0.01.
        total_iters (Optional[int]): The total number of iterations for the learning rate scheduler. Default is 5.
        early_stopping (Optional[bool]): Whether to use early stopping during training or not. Default is False.
        patience (Optional[int]): The number of epochs with no improvement after which training will be stopped. Default is 3.
        quality (Optional[str]): The metric to track for early stopping callback. Default is "val_loss".
        save_checkpoint (Optional[bool]): Whether to save checkpoints during training or not. Default is False.
        checkpoint_output_path (Optional[str]): The output path for saving checkpoints. Default is ".".
        save_top_k (Optional[int]): The number of best models to keep when saving checkpoints. Default is False.
        every_n_epochs (Optional[int]): Save a checkpoint every N epochs. Default is 3.
        log_mlflow (Optional[bool]): Whether to log training metrics using MLflow or not. Default is False.
    """
    model: WaveformModel
    loss: WaveformLoss
    sources: List[str]
    num_channels: int
    batch_size: Optional[int] = 4
    num_layers: Optional[int] = 3
    hidden_size: Optional[int] = 64
    num_workers: Optional[int] = 1
    bidirectional: Optional[bool] = True
    transfer_learning: Optional[bool] = True
    dropout: Optional[float] = 0.0
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
    save_top_k: Optional[int] = False
    every_n_epochs: Optional[int] = 3
    log_mlflow: Optional[bool] = False
