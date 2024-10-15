import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, NoReturn

import h5py
import librosa
import musdb
from tqdm import tqdm

from pipelines.utils import float32_to_int16
from pipelines.utils.process_audio import preprocess_audio
from pipelines.utils.hdf5 import write_single_audio_to_hdf5

SOURCE_TYPES = ["vocals", "drums", "bass", "other", "accompaniment"]

SAMPLE_RATE=44100
CHANNELS=2

dataset_dir = "datasets/musdb18"
subset="train"
split=""
hdf5s_dir=f"hdf5s/musdb18/sr={SAMPLE_RATE},chn={CHANNELS}/train"
sample_rate=SAMPLE_RATE
channels = CHANNELS


mono = True if channels == 1 else False
source_types = SOURCE_TYPES
resample_type = "kaiser_fast"

# Paths
os.makedirs(hdf5s_dir, exist_ok=True)

# Dataset of corresponding subset and split.
mus = musdb.DB(root=dataset_dir, subsets=[subset], split=split)
print("Subset: {}, Split: {}, Total pieces: {}".format(subset, split, len(mus)))

params = []  # A list of params for multiple processing.

for track_index in range(len(mus.tracks)):

    param = (
        dataset_dir,
        subset,
        split,
        track_index,
        source_types,
        mono,
        sample_rate,
        resample_type,
        hdf5s_dir,
    )

    params.append(param)


pack_hdf5s_time = time.time()


with ProcessPoolExecutor(max_workers=None) as pool:
    # Initialize tqdm progress bar with total equal to the length of params
    for _ in tqdm(pool.map(write_single_audio_to_hdf5, params), total=len(params)):
        pass

print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))