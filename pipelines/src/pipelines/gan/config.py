import os

import yaml
from pydantic import BaseModel


class TrainingGANConfig(BaseModel):
    model_prefix: str = "exp_musdb_1_wide_unpaired_ralsgan_4"
    n_iterations: int = 100000
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.9
    decay_lr: bool = False
    generator_batch_size_factor: int = 1
    n_critic: int = 5
    p_coeff: int = 10
    batch_size: int = 10
    noise_latent_dim: int = 100
    model_capacity_size: int = 64
    store_cost_every: int = 300
    progress_bar_step_iter_size: int = 400
    take_backup: bool = True
    backup_every_n_iters: int = 1000
    save_samples_every: int = 1000
    target_signals_dir: str = "mancini_piano/piano"
    other_signals_dir: str = "mancini_piano/noise"
    output_dir: str = "output"
    window_length: int = 16384
    sampling_rate: int = 16000
    normalize_audio: bool = True
    random_state: int = 2048


def load_gan_config(path: str) -> TrainingGANConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f'File: {path} does not exist.')
    with open(path, 'r') as data:
        yaml_data = yaml.safe_load(data)
    return TrainingGANConfig(**yaml_data)
