from typing import List, Optional

import lmdb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pipelines.gan import datanum_pb2
from pipelines.gan.config import TrainingGANConfig
from pipelines.gan.prepare.utils import get_map_size, get_sequence_with_singing_indices
from pipelines.gan.utils import get_recursive_files, load_wav


def write_lmdb(out_file_name: str, data_list: List[str], config: TrainingGANConfig) -> None:
    """
    Writes audio data to an LMDB database.

    Args:
        out_file_name (str): Path to the output LMDB file.
        data_list (List[str]): List of audio file paths to be stored in LMDB.
        config (TrainingGANConfig): Configuration object for GAN training.

    Notes:
        - Each entry in the LMDB contains:
          - Mixed audio data
          - Corresponding vocals data
          - Indices of segments with vocals
    """
    lmdb_output = lmdb.open(
        out_file_name,
        map_size=get_map_size(data_list, config=config)
    )
    with lmdb_output.begin(write=True) as txn:
        for audio_indx, audio_path in enumerate(tqdm(data_list)):
            if 'mixture' not in audio_path:
                continue  # Process only 'mixture' audio files

            # Load mixed and vocal audio data
            mixed_data = load_wav(audio_path, config=config).astype('float32')
            vocals_data = load_wav(
                audio_path.replace('mixture', 'vocals'),
                config=config
            ).astype('float32')

            # Get indices of vocal segments
            vocals_indices = get_sequence_with_singing_indices(vocals_data, window_size=800)

            # Create a protobuf data object
            datum = datanum_pb2.DataNum()
            datum.mixture = mixed_data.tobytes()
            datum.vocals = vocals_data.tobytes()
            datum.vocals_indices = vocals_indices.tobytes()

            # Serialize and store data in LMDB
            str_id = '{:08}'.format(audio_indx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


def create_lmdb(
    folder_name: str,
    out_file_name: str,
    is_train: bool = True,
    config: TrainingGANConfig = TrainingGANConfig()
) -> None:
    """
    Creates LMDB files for training and validation datasets.

    Args:
        folder_name (str): Path to the folder containing audio files.
        out_file_name (str): Path for the output LMDB file.
        is_train (bool, optional): Whether to create a training dataset. Default: True.
        config (TrainingGANConfig, optional): Configuration object for GAN training. Default: TrainingGANConfig().

    Notes:
        - If `is_train` is True, a validation set is also created using a 15% split from the training data.
        - The output files will have "_train" and "_valid" suffixes for training and validation datasets, respectively.
    """
    # Get all 'mixture.wav' files recursively
    audio_train = get_recursive_files(folder_name, 'mixture.wav')
    audio_valid: Optional[List[str]] = None

    if is_train:
        # Split data into training and validation sets
        audio_train, audio_valid = train_test_split(
            audio_train, test_size=0.15, random_state=config.random_state
        )

    # Write training data to LMDB
    write_lmdb(out_file_name, audio_train, config=config)

    if audio_valid:
        # Write validation data to LMDB
        valid_out_file_name = out_file_name.replace('_train', '') + '_valid'
        write_lmdb(valid_out_file_name, audio_valid, config=config)
