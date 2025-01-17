import lmdb
import numpy as np
import torch
import torch.utils

from pipelines.gan.config import TrainingGANConfig
from pipelines.gan.datanum_pb2 import DataNum
from pipelines.gan.utils import (
    create_stream_reader,
    get_recursive_files,
    numpy_to_tensor,
    sample_buffer,
)


class WavDataLoader():
    def __init__(self, folder_path, other_signals_folder, audio_extension='wav'):
        self.signal_paths = get_recursive_files(folder_path, audio_extension)
        self.mixed_wav_files = get_recursive_files(
            other_signals_folder, audio_extension)
        self.data_iter = None
        self.initialize_iterator()

    def initialize_iterator(self):
        data_iter = create_stream_reader(self.signal_paths, self.mixed_wav_files)
        self.data_iter = iter(data_iter)

    def __len__(self):
        return len(self.signal_paths)

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.data_iter)
        return (
            numpy_to_tensor(
                x['single']), numpy_to_tensor(
                x['mixed']), numpy_to_tensor(
                x['foreground']))


class LMDBWavLoader(torch.utils.data.Dataset):
    def __init__(self, config: TrainingGANConfig, lmdb_file_path, is_test=False):
        self.env = lmdb.open(lmdb_file_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.datum = DataNum()
        self.is_test = is_test
        self.config = config

    def __len__(self):
        n_entries = int(self.env.stat()['entries'])
        return n_entries

    def __getitem__(self, data_indx):
        index = None
        audio_indx = None
        index = data_indx
        with self.env.begin(write=False) as cursor:
            raw_datum = cursor.get('{:08}'.format(index).encode('ascii'))
        self.datum.ParseFromString(raw_datum)
        # float is represented by 4 bytes
        start_idx = None
        end_idx = None
        if self.is_test:
            return np.array(np.frombuffer(self.datum.vocals, dtype=np.float32)).reshape(
                -1), np.array(np.frombuffer(self.datum.mixture, dtype=np.float32)).reshape(-1)

        mixture, start_idx, end_idx = sample_buffer(
            self.datum.mixture, self.config, start_idx, end_idx)
        mixture = np.array(np.frombuffer(mixture, dtype=np.float32)).reshape(1, -1)
        vocals, _, _ = sample_buffer(self.datum.vocals, self.config, start_idx, end_idx)
        vocals = np.frombuffer(vocals, dtype=np.float32).reshape(1, -1)
        return vocals, mixture
