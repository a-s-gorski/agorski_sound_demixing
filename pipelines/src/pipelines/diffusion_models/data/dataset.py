import math
import librosa
import os
import numpy as np
from torch.utils.data import Dataset
import torch

from pipelines.diffusion_models.utils import _identity, get_duration_sec, load_audio


class MultiSourceDataset(Dataset):
    def __init__(self, sr, channels, min_duration, max_duration, aug_shift, sample_length, audio_files_dir, stems, transform=None):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration or math.ceil(sample_length / sr)
        self.max_duration = max_duration or math.inf
        self.sample_length = sample_length
        self.audio_files_dir = audio_files_dir
        self.stems = stems
        assert (
                sample_length / sr < self.min_duration
        ), f"Sample length {sample_length} per sr {sr} ({sample_length / sr:.2f}) should be shorter than min duration {self.min_duration}"
        self.aug_shift = aug_shift
        self.transform = transform if transform is not None else _identity
        self.init_dataset()

    def filter(self, tracks):
        # Remove files too short or too long
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(self.audio_files_dir, track)
            files = librosa.util.find_files(directory=f"{track_dir}", ext=["mp3", "opus", "m4a", "aac", "wav"])
            
            # skip if there are no sources per track
            if not files:
                continue
            
            durations_track = np.array([get_duration_sec(file, cache=True) * self.sr for file in files]) # Could be approximate
            
            # skip if there is a source that is shorter than minimum track length
            if (durations_track / self.sr < self.min_duration).any():
                continue
            
            # skip if there is a source that is longer than maximum track length
            if (durations_track / self.sr >= self.max_duration).any():
                continue
            
            # skip if in the track the different sources have different lengths
            if not (durations_track == durations_track[0]).all():
                print(f"{track} skipped because sources are not aligned!")
                print(durations_track)
                continue
            keep.append(track)
            durations.append(durations_track[0])
        
        print(f"self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")
        self.tracks = keep
        self.durations = durations
        self.cumsum = np.cumsum(np.array(self.durations))

    def init_dataset(self):
        # Load list of tracks and starts/durations
        tracks = os.listdir(self.audio_files_dir)
        print(f"Found {len(tracks)} tracks.")
        self.filter(tracks)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift  # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f"Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}"
        
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index]  # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        
        if offset > end - self.sample_length:  # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start:  # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert (
                start <= offset <= end - self.sample_length
        ), f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        
        offset = offset - start
        return index, offset

    def get_song_chunk(self, index, offset):
        track_name, total_length = self.tracks[index], self.durations[index]
        data_list = []
        for stem in self.stems:
            data, sr = load_audio(os.path.join(self.audio_files_dir, track_name, f'{stem}.wav'),
                                  sr=self.sr, offset=offset, duration=self.sample_length, approx=True)
            data = 0.5 * data[0:1, :] + 0.5 * data[1:, :]
            assert data.shape == (
                self.channels,
                self.sample_length,
            ), f"Expected {(self.channels, self.sample_length)}, got {data.shape}"
            data_list.append(data)
        return np.concatenate(data_list, axis=0)

    def get_item(self, item):
        index, offset = self.get_index_offset(item)
        wav = self.get_song_chunk(index, offset)
        return self.transform(torch.from_numpy(wav))

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)