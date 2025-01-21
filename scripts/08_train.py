import torch
import pathlib
import os

from pipelines.utils.train_script import train
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--workspace", type=str, required=False, help="workdir directory", default=".")
    parser.add_argument("--gpus", type=int, required=False, default=-1, help="how many gpus you want to use, -1 means none")
    parser.add_argument("--config_yaml", type=str, required=True)
    
    args = parser.parse_args()
    
    workspace = args.workspace
    gpus = args.gpus if torch.cuda.is_available() else -1
    config_yaml = args.config_yaml
    filename = pathlib.Path(os.getcwd()).stem

    train(
        workspace=workspace,
        gpus=gpus,
        config_yaml=config_yaml,
        filename=filename
    )
    

if __name__ == "__main__":
    main()