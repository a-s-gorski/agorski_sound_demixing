import logging
from typing import Dict, Tuple

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pipelines.dataset.base_dataset import BaseDataset
from pipelines.models.convtasnet import ConvTasNet
from pipelines.models.gan import UnetGAN
from pipelines.models.hdemucs import HDEMUCS
from pipelines.models.lstm import BiLSTM
from pipelines.models.vae import VAE
from pipelines.types.training_signal_enriched import TrainingGenerativeConfig
from pipelines.utils.train_signal_enriched import (
    compute_enriched,
    get_model,
    prepare_model_input,
    run_inference,
)
from pipelines.utils.training import get_trainer
from pipelines.utils.training_waveform import calculate_losses, get_signal_by_source


def compute_dataset_enriched(model: BiLSTM | ConvTasNet | HDEMUCS,
                             train_ds: BaseDataset,
                             val_ds: BaseDataset,
                             test_ds: BaseDataset) -> Tuple[BaseDataset,
                                                            BaseDataset,
                                                            BaseDataset]:
    """Compute the enriched datasets for training, validation, and testing using the specified model.

    Args:
        model (BiLSTM | ConvTasNet | HDEMUCS): The model used to compute the enriched datasets.
        train_ds (BaseDataset): Training dataset.
        val_ds (BaseDataset): Validation dataset.
        test_ds (BaseDataset): Testing dataset.

    Returns:
        Tuple[BaseDataset, BaseDataset, BaseDataset]: Enriched datasets for training, validation, and testing.
    """
    logger = logging.getLogger(__name__)

    logging.info("Computing enriched datasets.")
    train_enriched_ds = compute_enriched(model, train_ds)
    val_enriched_ds = compute_enriched(model, val_ds)
    test_enriched_ds = compute_enriched(model, test_ds)

    return train_enriched_ds, val_enriched_ds, test_enriched_ds


def train_generative_model(train_ds: BaseDataset,
                           val_ds: BaseDataset,
                           params: Dict[str,
                                        str | float]) -> VAE | UnetGAN:
    """Train a generative model (VAE or UnetGAN) using the provided training and validation datasets and configuration.

    Args:
        train_ds (BaseDataset): Training dataset.
        val_ds (BaseDataset): Validation dataset.
        params (Dict[str, str | float]): Dictionary containing configuration parameters for training.

    Returns:
        VAE | UnetGAN: Trained generative model (VAE or UnetGAN).
    """
    logger = logging.getLogger(__name__)

    config = TrainingGenerativeConfig(**params)

    logger.info("Preparing model input")
    train_ds, config, original_signal_len = prepare_model_input(
        config, train_ds)
    val_ds, _, _ = prepare_model_input(config, val_ds)

    logger.info("Loading model")
    model = get_model(config=config)

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

    torch.set_float32_matmul_precision('medium')

    if config.log_mlflow:
        with mlflow.start_run():
            logger.info("Logging using mlflow.")
            trainer = get_trainer(config=config)
            mlflow.pytorch.autolog()
            trainer.fit(model, train_dataloader, val_dataloader)
    else:
        logger.info("Logging using tensorboard.")
        trainer = get_trainer(config=config)
        trainer.fit(model, train_dataloader, val_dataloader)

    logger.info("Training finished. Saving model.")
    return model


def test_generative_model(model: UnetGAN | VAE,
                          test_ds: BaseDataset,
                          training_enriched_params: Dict[str,
                                                         str | float],
                          training_params: Dict[str,
                                                str | float],
                          processing_params: Dict[str,
                                                  str | float]) -> Tuple[pd.DataFrame,
                                                                         torch.Tensor]:
    """Test a generative model (VAE or UnetGAN) using the provided test dataset and configuration parameters.

    Args:
        model (UnetGAN | VAE): Trained generative model (VAE or UnetGAN).
        test_ds (BaseDataset): Test dataset.
        training_enriched_params (Dict[str, str | float]): Configuration parameters used for enriching the training data.
        training_params (Dict[str, str | float]): Configuration parameters used for training the generative model.
        processing_params (Dict[str, str | float]): Configuration parameters used for processing.

    Returns:
        Tuple[pd.DataFrame, torch.Tensor]: Tuple containing the evaluation metrics DataFrame and the output signal
        generated by the model for each source.
    """
    logger = logging.getLogger(__name__)

    config = TrainingGenerativeConfig(**training_enriched_params)
    logger.info("Preparing test dataset.")
    test_ds, _, _ = prepare_model_input(config, test_ds)

    print("X", test_ds.X.shape)
    logger.info("Running inference.")
    predicted = run_inference(model, test_ds)

    # need to extend dim since the signal is mono
    predicted = predicted.unsqueeze(-2)
    targets = test_ds.Y

    logger.info("Computing metrics.")
    metrics_dict = {}
    losses_dict = calculate_losses(
        config=config,
        predicted=predicted,
        targets=targets
    )
    metrics_dict.update(losses_dict)
    metrics_dict["processing_params"] = str(processing_params)
    metrics_dict["training_params"] = str(training_params)
    metrics_dict["training_generative_params"] = str(training_enriched_params)

    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    output_signal_by_sources = get_signal_by_source(
        config=config, signal=predicted
    )

    return metrics_df, output_signal_by_sources
