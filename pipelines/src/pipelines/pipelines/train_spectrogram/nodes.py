import logging
from typing import Dict

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pipelines.types.training_spectrogram import TrainingSpectrogramConfig
from pipelines.utils.training import get_trainer
from pipelines.utils.training_spectrogram import (
    convert_spectrogram_to_signal,
    get_model,
    get_predictions,
)
from pipelines.utils.training_waveform import calculate_losses


def train_spectrogram_node(train_ds, val_ds, params):
    """
    Trains a spectrogram model using the provided training and validation datasets.

    Parameters:
        train_ds (Dataset): Training dataset containing input spectrograms and corresponding targets.
        val_ds (Dataset): Validation dataset containing input spectrograms and corresponding targets.
        params (dict): A dictionary containing parameters for configuring the training process.

    Returns:
        torch.nn.Module: Trained model after completion of the training process.
    """
    logger = logging.getLogger(__name__)

    config = TrainingSpectrogramConfig(**params)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True)
    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False)

    model = get_model(config)
    logger.info("Setting set_float32_matmul_precision to medium for rtx3060.")
    torch.set_float32_matmul_precision('medium')

    if config.log_mlflow:
        with mlflow.start_run():
            logger.info("Starting training with mlflow logging.")
            trainer = get_trainer(config=config)
            mlflow.autolog()
            trainer.fit(model, train_dataloader, val_dataloader)
    else:
        logger.info("Starting training with local tensorboard logging.")
        trainer = get_trainer(config=config)
        trainer.fit(model, train_dataloader, val_dataloader)

    return model


def inference_spectrogram_node(model, test_ds) -> Dict[str, torch.Tensor]:
    """
    Performs inference using the provided model on the given test dataset.

    Parameters:
        model (torch.nn.Module): Trained model to use for inference.
        test_ds (Dataset): Test dataset containing input spectrograms and corresponding targets.

    Returns:
        dict: A dictionary containing the predicted and target tensors.
            Keys: "predicted" and "targets".
    """
    logger = logging.getLogger(__name__)

    logger.info("Running inference.")
    predicted = get_predictions(model=model, spectrograms=test_ds.X)
    return {"predicted": predicted, "targets": test_ds.Y}


def spectrogram_to_signal_node(
        freq_signal: Dict[str, torch.Tensor], params: Dict[str, str | float], original_sizes: Dict[str, int]):
    """
    Converts frequency domain signals back to time domain signals.

    Parameters:
        freq_signal (dict): A dictionary containing the predicted and target frequency domain signals.
            Keys: "predicted" and "targets", each holding torch.Tensor frequency representations.
        params (dict): A dictionary containing additional parameters required for the conversion process.
        original_sizes (dict): A dictionary containing original sizes of the signals.

    Returns:
        dict: A dictionary containing the converted time domain signals.
            Keys: "predicted" and "targets", each holding torch.Tensor time domain signals.
    """
    logger = logging.getLogger(__name__)

    predicted = freq_signal["predicted"]
    targets = freq_signal["targets"]

    logger.info(
        "Starting inverse tranform for converting spectrograms back to signals.")
    print("predicted_signal", predicted.shape, "org", original_sizes)
    predicted_signals = convert_spectrogram_to_signal(
        predicted, original_sizes)

    target_signals = convert_spectrogram_to_signal(targets, original_sizes)
    logger.info("Inverse transform computed successfully.")

    return {"predicted": predicted_signals, "targets": target_signals}


def test_signals(signals, params_signal, params_spectrogram, params_training):
    """
    Tests the quality of predicted signals against target signals and calculates performance metrics.

    Parameters:
        signals (dict): A dictionary containing the predicted and target time domain signals.
            Keys: "predicted" and "targets", each holding torch.Tensor time domain signals.
        params_signal (dict): Parameters related to the signals.
        params_spectrogram (dict): Parameters related to spectrogram conversion.
        params_training (dict): Parameters related to the training process.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the calculated performance metrics and relevant parameters.
    """
    logger = logging.getLogger(__name__)
    config = TrainingSpectrogramConfig(**params_training)

    logger.info("Computing loss metrics.")
    metrics_dict = {}

    losses_dict = calculate_losses(
        config=config,
        predicted=signals['predicted'].detach(),
        targets=signals['targets'].detach())

    metrics_dict.update(losses_dict)
    logger.info(
        "Metrics computed successfully. Starting to log training and transformation parameters.")
    metrics_dict["processing_params_signal"] = str(params_signal)
    metrics_dict["processing_params_spectrogram"] = str(params_spectrogram)
    metrics_dict["training_params"] = str(params_training)

    metrics_df = pd.DataFrame(metrics_dict, index=[0])

    return metrics_df
