from typing import Union
from pathlib import Path
import torch
import json
from pipelines.diffusion_models.separate.base_separation_dataset import SeparationDataset
from pipelines.diffusion_models.separate.model import Separator
from pipelines.diffusion_models.separate.separate_dataset import separate_dataset

@torch.no_grad()
def separate_slakh(
        output_dir: Union[str, Path],
        dataset: SeparationDataset,
        separator: Separator,
        num_steps: int = 150,
        batch_size: int = 16,
        resume: bool = False,
    ):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create chunks metadata
    chunk_data = []
    for i in range(len(dataset)):
        start_sample, end_sample = dataset.get_chunk_indices(i)
        chunk_data.append(
            {
                "chunk_index": i,
                "track": dataset.get_chunk_track(i),
                "start_chunk_sample": start_sample,
                "end_chunk_sample": end_sample,
                "track_sample_rate": dataset.sample_rate,
                "start_chunk_seconds": start_sample / dataset.sample_rate,
                "end_chunk_in_seconds": end_sample / dataset.sample_rate,
            }
        )

    # Save chunk metadata
    with open(output_dir / "chunk_data.json", "w") as f:
        json.dump(chunk_data, f, indent=1)

    # Separate chunks
    separate_dataset(
        dataset=dataset,
        separator=separator,
        save_path=output_dir,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume
    )
