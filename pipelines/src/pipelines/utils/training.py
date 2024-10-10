import os
from datetime import datetime

import mlflow
import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pipelines.types.training_signal_enriched import TrainingGenerativeConfig
from pipelines.types.training_spectrogram import TrainingSpectrogramConfig
from pipelines.types.training_waveform import TrainingWaveformConfig


def get_trainer(config: TrainingWaveformConfig |
                TrainingSpectrogramConfig | TrainingGenerativeConfig) -> pl.Trainer:
    """
    Helper function for getting pytorch-lightning based trainer

    Parameters:
        config: parsed configuration file with training parameters.

    Returns:
        trainer: pytorch-lightning based trainer with callbacks if specified.
    """
    callbacks = []
    if config.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=config.quality,
                patience=config.patience))

    if config.save_checkpoint:
        checkpoint_dir = f"{str(config.model)}_{str(datetime.now())}"
        checkpoint_output_path = os.path.join(
            config.checkpoint_output_path, checkpoint_dir)
        os.makedirs(checkpoint_output_path, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_output_path,
                save_top_k=config.save_top_k,
                every_n_epochs=config.every_n_epochs))

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1)

    if config.log_mlflow:
        active_run = mlflow.active_run()
        if not active_run:
            active_run = mlflow.start_run()
        exp_name = active_run.info.experiment_id
        run_id = active_run.info.run_id
        mlflow_logger = MLFlowLogger(experiment_name=exp_name, run_id=run_id)
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            strategy="auto",
            max_epochs=config.max_epochs,
            callbacks=callbacks,
            log_every_n_steps=1,
            logger=mlflow_logger)

    return trainer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
