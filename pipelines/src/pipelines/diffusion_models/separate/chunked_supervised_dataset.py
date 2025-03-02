from typing import Union, List, Optional, Tuple
from pathlib import Path
import torch

from pipelines.diffusion_models.separate.supervised_dataset import SupervisedDataset
from pipelines.diffusion_models.separate.utils import get_nonsilent_and_multi_instr_chunks

class ChunkedSupervisedDataset(SupervisedDataset):
    def __init__(
        self,
        audio_dir: Union[Path, str],
        stems: List[str],
        sample_rate: int,
        max_chunk_size: int,
        min_chunk_size: int,
        silence_threshold: Optional[float]= None,
        only_multisource: bool = False,
    ):
        super().__init__(audio_dir=audio_dir, stems=stems, sample_rate=sample_rate)

        self.max_chunk_size ,self.min_chunk_size= max_chunk_size, min_chunk_size
        self.available_chunk = {}
        self.index_to_track, self.index_to_chunk = [], []
        self.silence_threshold = silence_threshold
        self.only_multisource = only_multisource

    
        #with mp.Pool() as pool:
        for track in self.tracks:
            _, available_chunks = self._get_available_chunks(track)
            self.available_chunk[track] = available_chunks
            self.index_to_track.extend([track] * len(available_chunks))
            self.index_to_chunk.extend(available_chunks)

        assert len(self.index_to_chunk) == len(self.index_to_track)

    def __len__(self):
        return len(self.index_to_track)

    def get_chunk_track(self, item: int) -> str:
        return self.index_to_track[item]

    def get_chunk_indices(self, item: int) -> Tuple[int, int]:
        ci = self.index_to_chunk[item]
        return ci * self.max_chunk_size, (ci + 1) * self.max_chunk_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        chunk_start, chunk_stop = self.get_chunk_indices(item)
        tracks = self.get_tracks(self.get_chunk_track(item))
        tracks = tuple([t[:, chunk_start:chunk_stop] for t in tracks])
        return tracks
    
    def _get_available_chunks(self, track: str):
        tracks = self.get_tracks(track) # (num_stems, [1, num_samples])
        available_chunks = get_nonsilent_and_multi_instr_chunks(
            tracks, 
            self.max_chunk_size, 
            self.min_chunk_size,
            self.silence_threshold,
            self.only_multisource,
            )
        return track, available_chunks 