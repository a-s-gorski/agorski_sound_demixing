import logging
from typing import Dict

import mlflow
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from pipelines.dataset.base_dataset import BaseDataset
from pipelines.types.training_waveform import TrainingWaveformConfig, WaveformModel
from pipelines.utils.training import get_trainer
from pipelines.utils.training_waveform import (
    calculate_losses,
    convert_dataset_to_mono,
    convert_sources_signal_to_mono,
    convert_to_stereo,
    get_model,
    get_model_predictions,
    get_signal_by_source,
)


def train_model_node(train_ds: BaseDataset, val_ds: BaseDataset,
                     params: Dict[str, bool | int], random_state: int):
    """
    Trains a neural network model on the provided training dataset using the given parameters.

    Parameters:
        train_ds (BaseDataset): The training dataset containing input audio waveforms and corresponding targets.
        val_ds (BaseDataset): The validation dataset containing input audio waveforms and corresponding targets.
        params (Dict[str, bool | int]): A dictionary containing configuration parameters for training the model.
        random_state (int): Random seed for reproducibility.

    Returns:
        nn.Module: The trained neural network model.
    """

    logger = logging.getLogger(__name__)

    config = TrainingWaveformConfig(**params)

    torch.manual_seed(random_state)

    if config.model == WaveformModel.HDEMUCS and train_ds.X.shape[-2] == 1:
        logger.warning(
            "Detected mono signal while hdemucs is trained on stereo - converting")
        train_ds = convert_to_stereo(train_ds)
        val_ds = convert_to_stereo(val_ds)

    if config.model == WaveformModel.CONVTASNET and train_ds.X.shape[-2] == 2:
        logger.warning(
            "Detected stereo signal while convtasnet is trained on mono - converting")
        train_ds = convert_to_stereo(train_ds)
        val_ds = convert_to_stereo(val_ds)

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

    model = get_model(config=config)

    # for rtx 3060
    torch.set_float32_matmul_precision('medium')
    if config.log_mlflow:
        with mlflow.start_run():
            # trainer has to be executed here  as well as it needs to get
            # active run id and exp name
            trainer = get_trainer(config=config)
            mlflow.pytorch.autolog()
            trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer = get_trainer(config=config)
        trainer.fit(model, train_dataloader, val_dataloader)

    return model


def test_model_node(
        model: nn.Module,
        test_ds: BaseDataset,
        params_signal,
        params_train):
    """
    Tests the trained neural network model on the given test dataset and calculates evaluation metrics.

    Paramters:
        model (nn.Module): The trained neural network model.
        test_ds (BaseDataset): The test dataset containing input audio waveforms and corresponding targets.
        params_signal: (Not specified in the function. Please provide information about this argument.)
        params_train: (Not specified in the function. Please provide information about this argument.)

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing:
            - metrics_df (pd.DataFrame): A DataFrame containing evaluation metrics for the model's performance.
            - output_signal_by_sources (dict): A dictionary containing output signals separated by sources.
    """

    logger = logging.getLogger(__name__)

    config = TrainingWaveformConfig(**params_train)

    if config.model == WaveformModel.HDEMUCS and test_ds.X.shape[-2] == 1:
        logger.warning(
            "Detected mono signal while hdemucs is trained on stereo - converting")
        test_ds = convert_to_stereo(test_ds)

    predicted = get_model_predictions(model, test_ds).detach()
    if predicted.shape[-1] == 2:
        logging.info(
            "All test predictions are run for a single chanel - converting")
        predicted = convert_sources_signal_to_mono(predicted)

    test_ds = convert_dataset_to_mono(test_ds)
    targets = test_ds.Y

    metrics_dict = {}
    losses_dict = calculate_losses(
        config=config,
        predicted=predicted,
        targets=targets)
    metrics_dict.update(losses_dict)
    metrics_dict["processing_params"] = str(params_signal)
    metrics_dict["training_params"] = str(params_train)

    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    output_signal_by_sources = get_signal_by_source(
        config=config, signal=predicted)

    return metrics_df, output_signal_by_sources
