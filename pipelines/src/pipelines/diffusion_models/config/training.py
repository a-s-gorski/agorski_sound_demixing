from typing import List

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    in_channels: int = 4
    channels: int = 256
    patch_factor: int = 16
    patch_blocks: int = 1
    resnet_groups: int = 8
    kernel_multiplier_downsample: int = 2
    kernel_sizes_init: List[int] = [1, 3, 7]
    multipliers: List[int] = [1, 2, 4, 4, 4, 4, 4]
    factors: List[int] = [4, 4, 4, 2, 2, 2]
    num_blocks: List[int] = [2, 2, 2, 2, 2, 2]
    attentions: List[bool] = [False, False, False, True, True, True]
    attention_heads: int = 8
    attention_features: int = 128
    attention_multiplier: int = 2
    use_nearest_upsample: bool = False
    use_skip_scale: bool = True
    use_attention_bottleneck: bool = True
    diffusion_sigma_distribution_mean: float = -3.0
    diffusion_sigma_distribution_std: float = 1.0
    diffusion_sigma_data: float = 0.2
    diffusion_dynamic_threshold: float = 0.0
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
