import torch

from typing import Optional
from pathlib import Path

from pipelines.diffusion_models.separation.data.chunked_supervised_dataset import ChunkedSupervisedDataset
from pipelines.diffusion_models.separation.data.base_dataset import SeparationDataset

from torch.utils.data import DataLoader
from math import ceil

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



@torch.no_grad()
def separate(
        output_dir: Union[str, Path],
        dataset: SeparationDataset,
        separator: Separator,
        num_steps: int = 150,
        batch_size: int = 16,
        resume: bool = False,
    ):

    # output_dir = Path(output_dir)
    # output_dir.mkdir(exist_ok=True)

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

    # # Save chunk metadata
    # with open(output_dir / "chunk_data.json", "w") as f:
    #     json.dump(chunk_data, f, indent=1)

    # Separate chunks
    separate_dataset(
        dataset=dataset,
        separator=separator,
        save_path=output_dir,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume
    )

@torch.no_grad()
def separate_directory(
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
    # config = stringify(locals())
    # output_dir = Path(output_dir)

    dataset = ChunkedSupervisedDataset(
        audio_dir=dataset_path,
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=22050,
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
        stems=["bass", "drums", "guitar", "piano"],
        sigma_schedule=KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0),
        differential_fn=diff_fn,
        s_churn=s_churn,
        num_resamples=num_resamples,
        use_tqdm=True,
    )
        
    separate(
        output_dir=output_dir,
        dataset=dataset,
        separator=separator,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume,
    )
    