import torch
from typing import Optional
from pathlib import Path
import functools
from audio_diffusion_pytorch import KarrasSchedule
import yaml

from pipelines.diffusion_models.model import Model
from pipelines.diffusion_models.utils import stringify
from pipelines.diffusion_models.differential import differential_with_dirac, differential_with_gaussian
from pipelines.diffusion_models.separate.chunked_supervised_dataset import ChunkedSupervisedDataset
from pipelines.diffusion_models.separate.model import MSDMSeparator

@torch.no_grad()
def separate_slakh_msdm(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    num_resamples: int = 1,
    num_steps: int = 150,
    batch_size: int = 16,
    resume: bool = True,
    device: float = torch.device("cuda:0"),
    s_churn: float = 20.0,
    sigma_min: float = 1e-4,
    sigma_max: float = 1.0,
    use_gaussian: bool = False,
    source_id: Optional[int] = None,
    gamma: Optional[float] = None,
):
    config = stringify(locals())
    output_dir = Path(output_dir)

    dataset = ChunkedSupervisedDataset(
        audio_dir=dataset_path,
        stems=["bass", "drums", "mixture", "other"],
        sample_rate=44100,
        max_chunk_size=262144,
        min_chunk_size=262144,
    )

    model = Model.load_from_checkpoint(model_path).to(device)

    if use_gaussian:
        assert gamma is not None
        diff_fn = functools.partial(differential_with_gaussian, gamma_fn=lambda s: gamma * s)
    else:
        assert source_id is not None
        diff_fn = functools.partial(differential_with_dirac, source_id=source_id)
    
    separator = MSDMSeparator(
        model=model,
        stems=["bass", "drums", "mixture", "other"],
        sigma_schedule=KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0),
        differential_fn=diff_fn,
        s_churn=s_churn,
        num_resamples=num_resamples,
        use_tqdm=True,
    )
        
    separate_slakh(
        output_dir=output_dir,
        dataset=dataset,
        separator=separator,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume,
    )
    
    with open(output_dir/"config.yaml", "w") as f:
        yaml.dump(config, f)