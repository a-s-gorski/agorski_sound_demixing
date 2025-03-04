from torch.utils.tensorboard import SummaryWriter
import torch
import json
import os
from datetime import datetime

from pipelines.gan.model.cycle_gan import CycleGan
from pipelines.gan.config import load_gan_config
from pipelines.gan.data.loader import LMDBWavLoader
from pipelines.gan.evaluate.metrics import calculate_metrics

config = load_gan_config('configs/gan/config.yaml')
writer = SummaryWriter(log_dir="./logs")

train_dataset = LMDBWavLoader(config, 'musdb_train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                        shuffle=True, num_workers=2,drop_last=True,pin_memory=True)
val_dataset = LMDBWavLoader(config, 'musdb_valid')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size,
                                        shuffle=True, num_workers=2,drop_last=True,pin_memory=True)

gan_model = CycleGan(train_loader=train_loader,val_loader=val_loader, config=config, validate=True, writer=writer)

# gan_model.train()

test_dataset = LMDBWavLoader(config=config, lmdb_file_path='musdb_test',is_test=True)

results = calculate_metrics(test_dataset=test_dataset, gan_model=gan_model, config=config)

os.makedirs("gan-results")
with open(os.path.join("gan-results", f"gan-results-{datetime.now()}.json"), "w") as f:
    json.dump(results, f, indent=4)