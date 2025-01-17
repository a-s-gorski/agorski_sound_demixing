import argparse
from pytorch_lightning import Trainer
from audio_diffusion_pytorch import LogNormalDistribution
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary, ModelCheckpoint
import os
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger

from pipelines.utils import read_yaml
from pipelines.diffusion_models.configs.dataset import DatasetConfig
from pipelines.diffusion_models.configs.model import ModelConfig
from pipelines.diffusion_models.model import Model as SeparationModel
from pipelines.diffusion_models.data.dataset import MultiSourceDataset
from pipelines.diffusion_models.data.dataloader import DatamoduleWithValidation


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the dataset configuration file."
    )
    
    parser.add_argument(
        "--training_config",
        type=str,
        required=True,
        help="Path to the training configuration file."
    )
    
    parser.add_argument(
        "--training_data_path",
        type=str,
        required=True,
        help="Path to training data."
    )
    
    parser.add_argument(
        "--validation_data_path",
        type=str,
        required=True,
        help="Path to validation data."
    )
    
    
    args = parser.parse_args()
    
    
    yaml_dataset_config = read_yaml(args.dataset_config)
    yaml_training_config = read_yaml(args.training_config)
    
    dataset_config = DatasetConfig(**yaml_dataset_config)
    training_config = ModelConfig(**yaml_training_config)
    
    print(args.training_data_path)
    print(args.validation_data_path)
    
    train_ds = MultiSourceDataset(**dataset_config.model_dump(),
                                  audio_files_dir=args.training_data_path)
    
    val_ds = MultiSourceDataset(**dataset_config.model_dump(),
                                audio_files_dir=args.validation_data_path)
    
    
    datamodule = DatamoduleWithValidation(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    training_model_dict = training_config.model_dump()
    training_model_dict.pop('diffusion_sigma_distribution_mean')
    training_model_dict.pop('diffusion_sigma_distribution_std')
    training_model_dict.pop('batch_size')
    training_model_dict.pop('num_workers')
    training_model_dict.pop('pin_memory')

    print(training_model_dict['num_blocks'])
    
    model = SeparationModel(**training_model_dict,
                            diffusion_sigma_distribution=LogNormalDistribution(
                                mean=training_config.diffusion_sigma_distribution_mean,
                                std=training_config.diffusion_sigma_distribution_std,
                            )
                            )
    
    
    
    
    callbacks = [
        ModelCheckpoint(
            monitor="valid_loss",
            save_top_k=1,
            save_last=False,
            mode="min",
            verbose=False,
            dirpath=f"{os.getcwd()}/diffusion_models/ckpts/{str(datetime.now())}"
        ),
        RichModelSummary(max_depth=2),
        RichProgressBar()
    ]
    
    logger = TensorBoardLogger(save_dir="./diffusion_logs")
    
    trainer = Trainer(precision="bf16", min_epochs=0, max_epochs=-1, enable_model_summary=True,
                      accelerator='gpu', devices=1,
                      log_every_n_steps=10, check_val_every_n_epoch=None, val_check_interval=4000,
                      callbacks=callbacks, logger=logger)
    
    trainer.fit(model=model, datamodule=datamodule)
    
    