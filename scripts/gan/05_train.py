from torch.utils.tensorboard import SummaryWriter
import torch

from pipelines.gan.model.cycle_gan import CycleGan
from pipelines.gan.config import load_gan_config
from pipelines.gan.data.loader import LMDBWavLoader

config = load_gan_config('configs/gan/config.yaml')
print("CONFIG", config)
print("CUDA", torch.cuda.is_available())
writer = SummaryWriter(log_dir="./logs")

train_dataset = LMDBWavLoader(config, 'musdb_train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                        shuffle=True, num_workers=2,drop_last=True,pin_memory=True)
val_dataset = LMDBWavLoader(config, 'musdb_valid')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size,
                                        shuffle=True, num_workers=2,drop_last=True,pin_memory=True)

gan_model = CycleGan(train_loader=train_loader,val_loader=val_loader, config=config, validate=True, writer=writer)
gan_model.train()