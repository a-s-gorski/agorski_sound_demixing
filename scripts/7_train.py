import torch
import pathlib
import os

from pipelines.utils.train_script import train


workspace = "."
gpus=1 if torch.cuda.is_available() else -1
config_yaml="./configs/training/vocals-accompaniment,resunet_subbandtime.yaml"
filename = pathlib.Path(os.getcwd()).stem

train(
    workspace=workspace,
    gpus=gpus,
    config_yaml=config_yaml,
    filename=filename
)