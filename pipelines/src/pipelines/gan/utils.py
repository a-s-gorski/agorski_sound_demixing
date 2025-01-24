import logging
import os
from typing import Generator, List, Optional, Tuple

import librosa
import numpy as np
import pescador
import soundfile
import torch
from scipy.io.wavfile import read as wavread

from pipelines.gan.config import TrainingGANConfig

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_recursive_files(folder_path: str, ext: str) -> List[str]:
    """Recursively retrieves files with a given extension from a folder.

    Args:
        folder_path (str): Path to the folder to search.
        ext (str): File extension to look for.

    Returns:
        List[str]: List of file paths with the specified extension.
    """
    results = os.listdir(folder_path)
    out_files = []
    for file in results:
        if os.path.isdir(os.path.join(folder_path, file)):
            out_files += get_recursive_files(os.path.join(folder_path, file), ext)
        elif file.endswith(ext):
            out_files.append(os.path.join(folder_path, file))

    return out_files


def make_path(output_path: str) -> str:
    """Creates a directory if it does not exist.

    Args:
        output_path (str): Path to the directory to create.

    Returns:
        str: The created or existing directory path.
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


def sample_audio(
    audio_data: np.ndarray,
    config: TrainingGANConfig,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None
) -> Tuple[np.ndarray, int, int]:
    """Samples a window of audio data.

    Args:
        audio_data (np.ndarray): The input audio data.
        config (TrainingGANConfig): Configuration for training GAN.
        start_idx (Optional[int]): Start index of the sample. Random if None.
        end_idx (Optional[int]): End index of the sample. Random if None.

    Returns:
        Tuple[np.ndarray, int, int]: Sampled audio data, start index, and end index.
    """
    audio_len = len(audio_data)
    if audio_len == config.window_length:
        sample = audio_data
    else:
        if start_idx is None or end_idx is None:
            start_idx = np.random.randint(0, (audio_len - config.window_length) // 2)
            end_idx = start_idx + config.window_length
        sample = audio_data[start_idx:end_idx]
    sample = sample.astype('float32')
    assert not np.any(np.isnan(sample))
    return sample, start_idx, end_idx


def create_stream_reader(
    single_signal_file_list: List[str],
    other_signal_file_list: List[str],
    config: TrainingGANConfig
) -> Generator:
    """Creates a data stream reader for training.

    Args:
        single_signal_file_list (List[str]): List of primary audio file paths.
        other_signal_file_list (List[str]): List of secondary audio file paths.
        config (TrainingGANConfig): Configuration for training GAN.

    Returns:
        Generator: A generator yielding batches of mixed and single signals.
    """
    data_streams = []
    other_signal_len = len(other_signal_file_list)
    for audio_path in single_signal_file_list:
        other_signal_idx = np.random.randint(0, other_signal_len)
        stream = pescador.Streamer(
            wav_generator,
            audio_path,
            other_signal_file_list[other_signal_idx]
        )
        data_streams.append(stream)
    mux = pescador.ShuffledMux(data_streams)
    batch_gen = pescador.buffer_stream(mux, config.batch_size)
    return batch_gen


def wav_generator(
        file_path: str, mixing_signal_path: str) -> Generator[dict, None, None]:
    """Generates audio samples and their mixtures.

    Args:
        file_path (str): Path to the primary audio file.
        mixing_signal_path (str): Path to the mixing audio file.

    Yields:
        dict: A dictionary containing single and mixed audio samples.
    """
    audio_data = load_wav(file_path)
    mixing_data = load_wav(mixing_signal_path)
    while True:
        sample, start_idx, end_idx = sample_audio(audio_data)
        mixing_sample, _, _ = sample_audio(mixing_data, start_idx, end_idx)

        mixing_ratio = np.random.uniform(0, 1)
        mixed_signal = mixing_ratio * mixing_sample + (1 - mixing_ratio) * sample

        yield {'single': sample, 'mixed': mixed_signal}


def audio_generator(
    audio_data: np.ndarray,
    config: TrainingGANConfig
) -> Generator[np.ndarray, None, None]:
    """Generates audio segments from an audio file.

    Args:
        audio_data (np.ndarray): The input audio data.
        config (TrainingGANConfig): Configuration for training GAN.

    Yields:
        np.ndarray: A segment of audio data.
    """
    audio_len = len(audio_data)
    n_iters = audio_len // config.window_length
    for i in range(n_iters + 1):
        start_idx = i * config.window_length
        end_idx = start_idx + config.window_length
        result = np.zeros(config.window_length)
        audio_size = audio_data[start_idx:end_idx].shape[0]
        result[:audio_size] = audio_data[start_idx:end_idx]
        yield result


def load_wav(
    wav_file_path: str,
    config: TrainingGANConfig,
    fast_loading: bool = False
) -> np.ndarray:
    """Loads a WAV file into a numpy array.

    Args:
        wav_file_path (str): Path to the WAV file.
        config (TrainingGANConfig): Configuration for training GAN.
        fast_loading (bool): Whether to use faster loading with scipy.

    Returns:
        np.ndarray: The loaded audio data.
    """
    try:
        if fast_loading:
            file_sampling_rate, audio_data = wavread(wav_file_path)
            if file_sampling_rate is not None and config.sampling_rate != file_sampling_rate:
                raise NotImplementedError('Scipy cannot resample audio.')
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.
            elif audio_data.dtype == np.float32:
                audio_data = np.copy(audio_data)
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
        else:
            audio_data, _ = librosa.load(wav_file_path, sr=config.sampling_rate)

        if config.normalize_audio:
            max_mag = np.max(np.abs(audio_data))
            if max_mag > 1:
                audio_data /= max_mag
    except Exception as e:
        logger.error(f"Could not load {wav_file_path}: {str(e)}")
        raise e

    audio_len = len(audio_data)
    if audio_len < config.window_length:
        pad_length = config.window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')

    return audio_data.astype('float32')


def save_samples(
    epoch_samples: List[np.ndarray],
    epoch: int,
    config: TrainingGANConfig,
    prefix: str = ''
) -> None:
    """Saves audio samples to disk.

    Args:
        epoch_samples (List[np.ndarray]): List of audio samples.
        epoch (int): Current epoch number.
        config (TrainingGANConfig): Configuration for training GAN.
        prefix (str): Prefix for file names.
    """
    sample_dir = make_path(os.path.join(config.output_dir, str(epoch)))

    for idx, sample in enumerate(epoch_samples):
        output_path = os.path.join(sample_dir, f"{prefix}_{idx + 1}.wav")
        soundfile.write(
            file=output_path,
            data=sample[0],
            samplerate=config.sampling_rate)


def sample_buffer(
    buffer_data: np.ndarray,
    config: TrainingGANConfig,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None
) -> Tuple[np.ndarray, int, int]:
    """Samples a segment from a buffer of audio data.

    Args:
        buffer_data (np.ndarray): The buffer containing audio data.
        config (TrainingGANConfig): Configuration for training GAN.
        start_idx (Optional[int]): Start index of the sample. Random if None.
        end_idx (Optional[int]): End index of the sample. Random if None.

    Returns:
        Tuple[np.ndarray, int, int]: Sampled audio data, start index, and end index.
    """
    audio_len = len(buffer_data) // 4
    if audio_len == config.window_length:
        sample = buffer_data
    else:
        if start_idx is None or end_idx is None:
            start_idx = np.random.randint(0, (audio_len - config.window_length) // 2)
            end_idx = start_idx + config.window_length
        sample = buffer_data[start_idx * 4:end_idx * 4]
    return sample, start_idx, end_idx


def numpy_to_tensor(numpy_array: np.ndarray) -> torch.Tensor:
    """Converts a numpy array to a PyTorch tensor.

    Args:
        numpy_array (np.ndarray): The input numpy array.

    Returns:
        torch.Tensor: The converted tensor.
    """
    numpy_array = numpy_array[:, np.newaxis, :]
    return torch.Tensor(numpy_array).to(device)


def sample_noise(size: int, config: TrainingGANConfig) -> torch.Tensor:
    """Generates random noise for GAN training.

    Args:
        size (int): Number of noise samples.
        config (TrainingGANConfig): Configuration for training GAN.

    Returns:
        torch.Tensor: A tensor of random noise.
    """
    z = torch.FloatTensor(size, config.noise_latent_dim).to(device)
    z.data.normal_()
    return z
