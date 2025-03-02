import torch
from pathlib import Path
from torch.utils.data import DataLoader
from math import ceil

from pipelines.diffusion_models.separate.model import Separator
from pipelines.diffusion_models.separate.base_separation_dataset import SeparationDataset
from pipelines.diffusion_models.separate.utils import save_separation

@torch.no_grad()
def separate_dataset(
    dataset: SeparationDataset,
    separator: Separator,
    num_steps: int,
    save_path: str,
    resume: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
):
    
    save_path = Path(save_path)
    if not resume and save_path.exists() and not len(list(save_path.glob("*"))) == 0:
        raise ValueError(f"Path {save_path} already exists!")

    # Get samples
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    # Main separation loop
    chunk_id = 0
    for batch_idx, batch in enumerate(loader):
        last_chunk_batch_id = chunk_id + batch[0].shape[0] - 1
        chunk_path = save_path / f"{last_chunk_batch_id}"
        
        if chunk_path.exists():
            print(f"Skipping path: {chunk_path}")
            chunk_id = chunk_id + batch[0].shape[0]
            continue

        print(f"{chunk_id=}")
        tracks = [b for b in batch]
        print(f"batch {batch_idx+1} out of {ceil(len(dataset) / batch[0].shape[0])}")
        
        # Generate mixture
        mixture = sum(tracks)
        seps_dict = separator.separate(mixture=mixture, num_steps=num_steps)

        # Save separated audio
        num_samples = tracks[0].shape[0]
        for i in range(num_samples):
            chunk_path = save_path / f"{chunk_id}"
            chunk_path.mkdir(parents=True, exist_ok=True)
            
            save_separation(
                separated_tracks={stem: sep[i] for stem, sep in seps_dict.items()},
                sample_rate=dataset.sample_rate,
                chunk_path=chunk_path,
            )
            chunk_id += 1