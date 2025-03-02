from pipelines.diffusion_models.separate.base_separation_dataset import SeparationDataset
from typing import Union, List, Tuple
import torch
import os
import functools
from pathlib import Path
import torchaudio
import itertools
import warnings

class SupervisedDataset(SeparationDataset):
    def __init__(
        self,
        audio_dir: Union[str, Path],
        stems: List[str],
        sample_rate: int,
        sample_eps_in_sec: int = 0.1
    ):
        super().__init__()
        self.sr = sample_rate
        self.sample_eps = round(sample_eps_in_sec * sample_rate)

        # Load list of files and starts/durations
        self.audio_dir = Path(audio_dir)
        self.tracks = sorted(os.listdir(self.audio_dir))
        self.stems = stems
        
        #TODO: add check if stem is never present in any track

    def __len__(self):
        return len(self.filenames)

    @functools.lru_cache(1)
    def get_tracks(self, track: str) -> Tuple[torch.Tensor, ...]:
        assert track in self.tracks
        stem_paths = {stem: self.audio_dir / track / f"{stem}.wav" for stem in self.stems}
        stem_paths = {stem: stem_path for stem, stem_path in stem_paths.items() if stem_path.exists()}
        assert len(stem_paths) >= 1, track
        
        stems_tracks = {}
        for stem, stem_path in stem_paths.items():
            audio_track, sr = torchaudio.load(stem_path)
            assert sr == self.sample_rate, f"sample rate {sr} is different from target sample rate {self.sample_rate}"
            stems_tracks[stem] = audio_track
                        
        channels, samples = zip(*[t.shape for t in stems_tracks.values()])
        
        for s1, s2 in itertools.product(samples, samples):
            assert abs(s1 - s2) <= self.sample_eps, f"{track}: {abs(s1 - s2)}"
            if s1 != s2:
                warnings.warn(
                    f"The tracks with name {track} have a different number of samples ({s1}, {s2})"
                )

        n_samples = min(samples)
        n_channels = channels[0]
        stems_tracks = {s:t[:, :n_samples] for s,t in stems_tracks.items()}
        
        for stem in self.stems:
            if not stem in stems_tracks:
                stems_tracks[stem] = torch.zeros(n_channels, n_samples)
        
        return tuple([stems_tracks[stem] for stem in self.stems])

    @property
    def sample_rate(self) -> int:
        return self.sr

    def __getitem__(self, item):
        return self.get_tracks(self.tracks[item])