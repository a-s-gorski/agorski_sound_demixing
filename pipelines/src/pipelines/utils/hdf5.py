import os
from typing import List, NoReturn

import h5py
import musdb
import numpy as np

from pipelines.utils import float32_to_int16
from pipelines.utils.process_audio import preprocess_audio


def write_single_audio_to_hdf5(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""
    (
        dataset_dir,
        subset,
        split,
        track_index,
        source_types,
        mono,
        sample_rate,
        resample_type,
        hdf5s_dir,
    ) = param

    # Dataset of corresponding subset and split.
    mus = musdb.DB(root=dataset_dir, subsets=[subset], split=split)
    track = mus.tracks[track_index]

    # Path to write out hdf5 file.
    hdf5_path = os.path.join(hdf5s_dir, "{}.h5".format(track.name))

    with h5py.File(hdf5_path, "w") as hf:

        hf.attrs.create("audio_name", data=track.name.encode(), dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        for source_type in source_types:

            audio = track.targets[source_type].audio.T
            # (channels_num, audio_samples)

            # Preprocess audio to mono / stereo, and resample.
            audio = preprocess_audio(
                audio, mono, track.rate, sample_rate, resample_type
            )
            # (channels_num, audio_samples) | (audio_samples,)

            hf.create_dataset(
                name=source_type, data=float32_to_int16(audio), dtype=np.int16
            )

        # Mixture
        audio = track.audio.T
        # (channels_num, audio_samples)

        # Preprocess audio to mono / stereo, and resample.
        audio = preprocess_audio(audio, mono, track.rate, sample_rate, resample_type)
        # (channels_num, audio_samples)

        hf.create_dataset(name="mixture", data=float32_to_int16(audio), dtype=np.int16)

    print("{} Write to {}, {}".format(track_index, hdf5_path, audio.shape))
