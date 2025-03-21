import math
import os

import av
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def get_duration_sec(file, cache=False):
    """
    Get the duration of an audio file in seconds.

    Args:
        file (str): Path to the audio file.
        cache (bool): If True, caches the duration in a separate file.

    Returns:
        float: Duration of the audio file in seconds.
    """
    if not os.path.exists(f"{file}.dur"):
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + ".dur", "w") as f:
                f.write(str(duration) + "\n")
        return duration

    with open(file + ".dur", "r") as f:
        duration = float(f.readline().strip("\n"))
    return duration


def load_audio(
        file,
        sr,
        offset,
        duration,
        resample=True,
        approx=False,
        time_base="samples",
        check_duration=True):
    """
    Load an audio file with specified sampling rate, offset, and duration.

    Args:
        file (str): Path to the audio file.
        sr (int): Target sampling rate.
        offset (int): Start offset in samples or seconds.
        duration (int): Number of samples or seconds to load.
        resample (bool): Whether to resample the audio to the target sampling rate.
        approx (bool): If True, allows minor shifts to fit within audio duration.
        time_base (str): Unit of `offset` and `duration` ("samples" or "sec").
        check_duration (bool): If True, ensures the requested duration is available.

    Returns:
        tuple: (numpy.ndarray of shape (channels, samples), sampling rate)
    """
    resampler = None
    if time_base == "sec":
        offset = offset * sr
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    if not os.path.exists(file):
        return np.zeros((2, duration), dtype=np.float32), sr
    container = av.open(file)
    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    if approx:

        if offset + duration > audio_duration * sr:
            # Move back one window. Cap at audio_duration
            offset = min(audio_duration * sr - duration, offset - duration)
    else:
        if check_duration:
            assert (
                offset + duration <= audio_duration * sr
            ), f"End {offset + duration} beyond duration {audio_duration*sr}"
    if resample:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
    else:
        assert sr == audio.sample_rate
    offset = int(
        offset / sr / float(audio.time_base)
    )  # int(offset / float(audio.time_base)) # Use units of time_base for seeking
    # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    duration = int(duration)
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    for frame in container.decode(audio=0):  # Only first audio stream
        if resample:
            frame.pts = None
            frame = resampler.resample(frame)
        frame = frame[0].to_ndarray(format="fltp")  # Convert to floats and not int16
        read = frame.shape[-1]
        if total_read + read > duration:
            read = duration - total_read
        sig[:, total_read: total_read + read] = frame[:, :read]
        total_read += read
        if total_read == duration:
            break
    assert total_read <= duration, f"Expected {duration} frames, got {total_read}"
    return sig, sr


def _identity(x):
    return x


class MultiSourceDataset(Dataset):
    def __init__(
            self,
            sr,
            channels,
            min_duration,
            max_duration,
            aug_shift,
            sample_length,
            audio_files_dir,
            stems,
            transform=None):
        """
        A PyTorch dataset for loading multi-source audio tracks.

        Args:
            sr (int): Sampling rate.
            channels (int): Number of audio channels.
            min_duration (float): Minimum track duration in seconds.
            max_duration (float): Maximum track duration in seconds.
            aug_shift (bool): Whether to apply time shifts for data augmentation.
            sample_length (int): Number of samples per training example.
            audio_files_dir (str): Directory containing audio tracks.
            stems (list of str): List of audio source names (e.g., ["vocals", "drums"]).
            transform (callable, optional): Function to apply transformations to the data.
        """
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
        """
        Filter tracks based on duration constraints and alignment.

        Args:
            tracks (list of str): List of track names.

        Returns:
            None
        """
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(self.audio_files_dir, track)
            files = librosa.util.find_files(
                directory=f"{track_dir}", ext=[
                    "mp3", "opus", "m4a", "aac", "wav"])

            if not files:
                continue

            durations_track = np.array(
                [get_duration_sec(file, cache=True) * self.sr for file in files])  # Could be approximate

            if (durations_track / self.sr < self.min_duration).any():
                continue

            if (durations_track / self.sr >= self.max_duration).any():
                continue

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
        """
        Initialize the dataset by finding available tracks.

        Returns:
            None
        """
        tracks = os.listdir(self.audio_files_dir)
        print(f"Found {len(tracks)} tracks.")
        self.filter(tracks)

    def get_index_offset(self, item):
        """
        Compute the track index and offset within the track for a given dataset index.

        Args:
            item (int): Dataset index.

        Returns:
            tuple: (track index, offset within track)
        """
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length // 2
        shift = np.random.randint(-half_interval,
                                  half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift  # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f"Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}"

        # index <-> midpoint of interval lies in this song
        index = np.searchsorted(self.cumsum, midpoint)
        # start and end of current song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index]
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"

        if offset > end - self.sample_length:  # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start:  # Going under song
            offset = min(
                end - self.sample_length,
                offset + half_interval)  # Now should fit
        assert (
            start <= offset <= end - self.sample_length
        ), f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"

        offset = offset - start
        return index, offset

    def get_song_chunk(self, index, offset):
        """
        Load a segment of an audio track.

        Args:
            index (int): Track index.
            offset (int): Offset within the track.

        Returns:
            numpy.ndarray: Audio segment.
        """
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
