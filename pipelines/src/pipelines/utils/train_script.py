import logging
from functools import partial
from typing import NoReturn

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import DDPPlugin

from pipelines.callbacks import get_callbacks
from pipelines.data_processing.batch_data_preprocessor import (
    get_batch_data_preprocessor_class,
)
from pipelines.dataset.data_module import get_data_module
from pipelines.loss import get_loss_function
from pipelines.models import LitSourceSeparation, get_model_class
from pipelines.models.optimizers import get_lr_lambda
from pipelines.utils import check_configs_gramma, read_yaml
from pipelines.utils.dirs import get_dirs


def train(workspace, gpus, config_yaml, filename) -> NoReturn:
    r"""Train & evaluate and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int
        config_yaml: str, path of config file for training
    """

    num_workers = 8
    distributed = True if gpus > 1 else False
    evaluate_device = "cuda" if gpus > 0 else "cpu"

    # Read config file.
    configs = read_yaml(config_yaml)
    check_configs_gramma(configs)
    task_name = configs['task_name']
    input_source_types = configs['train']['input_source_types']
    target_source_types = configs['train']['target_source_types']
    input_channels = configs['train']['input_channels']
    output_channels = configs['train']['output_channels']
    batch_data_preprocessor_type = configs['train']['batch_data_preprocessor']
    model_type = configs['train']['model_type']
    loss_type = configs['train']['loss_type']
    optimizer_type = configs['train']['optimizer_type']
    learning_rate = float(configs['train']['learning_rate'])
    precision = configs['train']['precision']
    early_stop_steps = configs['train']['early_stop_steps']
    warm_up_steps = configs['train']['warm_up_steps']
    reduce_lr_steps = configs['train']['reduce_lr_steps']
    resume_checkpoint_path = configs['train']['resume_checkpoint_path']

    target_sources_num = len(target_source_types)

    # paths
    checkpoints_dir, logs_dir, logger, statistics_path = get_dirs(
        workspace, task_name, filename, config_yaml, gpus
    )

    # training data module
    data_module = get_data_module(
        workspace=workspace,
        config_yaml=config_yaml,
        num_workers=num_workers,
        distributed=distributed,
    )

    # batch data preprocessor
    BatchDataPreprocessor = get_batch_data_preprocessor_class(
        batch_data_preprocessor_type=batch_data_preprocessor_type
    )

    batch_data_preprocessor = BatchDataPreprocessor(
        input_source_types=input_source_types, target_source_types=target_source_types
    )

    # model
    # print("model_type", model_type)
    # return
    Model = get_model_class(model_type=model_type)
    model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        target_sources_num=target_sources_num,
    )

    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        logging.info(
            "Load pretrained checkpoint from {}".format(resume_checkpoint_path)
        )

    # loss function
    loss_function = get_loss_function(loss_type=loss_type)

    # callbacks
    callbacks = get_callbacks(
        task_name=task_name,
        config_yaml=config_yaml,
        workspace=workspace,
        checkpoints_dir=checkpoints_dir,
        statistics_path=statistics_path,
        logger=logger,
        model=model,
        evaluate_device=evaluate_device,
    )
    # callbacks = []

    # learning rate reduce function
    lr_lambda = partial(
        get_lr_lambda, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps
    )

    # pytorch-lightning model
    pl_model = LitSourceSeparation(
        batch_data_preprocessor=batch_data_preprocessor,
        model=model,
        optimizer_type=optimizer_type,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda=lr_lambda,
    )

    # trainer
    trainer = pl.Trainer(
        checkpoint_callback=False,
        gpus=gpus,
        callbacks=callbacks,
        max_steps=early_stop_steps,
        accelerator="ddp",
        sync_batchnorm=True,
        precision=precision,
        replace_sampler_ddp=False,
        plugins=[DDPPlugin(find_unused_parameters=False)],
        profiler='simple',
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(pl_model, data_module)


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="")
#     subparsers = parser.add_subparsers(dest="mode")

#     parser_train = subparsers.add_parser("train")
#     parser_train.add_argument(
#         "--workspace", type=str, required=True, help="Directory of workspace."
#     )
#     parser_train.add_argument("--gpus", type=int, required=True)
#     parser_train.add_argument(
#         "--config_yaml",
#         type=str,
#         required=True,
#         help="Path of config file for training.",
#     )

#     args = parser.parse_args()
#     args.filename = pathlib.Path(__file__).stem

#     if args.mode == "train":
#         train(args)

#     else:
#         raise Exception("Error argument!")
