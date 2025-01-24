import numpy as np

from pipelines.gan.config import TrainingGANConfig
from pipelines.gan.utils import load_wav


def get_map_size(files: list[str], config: TrainingGANConfig) -> int:
    """
    Calculates the size of a map based on the provided audio files and configuration.

    Args:
        files (list[str]): List of file paths to audio files.
        config (TrainingGANConfig): Configuration object for the GAN training.

    Returns:
        int: The calculated map size.
    """
    return load_wav(files[0], config=config).nbytes * 10 * (len(files) + 2)


def get_silent_set(input_audio: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Identifies contiguous silent segments in an audio array.

    Args:
        input_audio (np.ndarray): Array of audio samples.

    Returns:
        list[tuple[int, int, int]]: A list of tuples containing the start index, end index, and length of each silent segment.
    """
    indices = np.where(input_audio == 0)[0]
    index_sets = []
    window = 16384
    counter = 0
    prev_index = -1
    first_index = -1

    for index in indices:
        if counter == 0:
            first_index = index
        if index - prev_index == 1:
            counter += 1
        else:
            if counter > window:
                index_sets.append((first_index, prev_index, counter))
            counter = 0
        prev_index = index

    if counter > window:
        index_sets.append((first_index, prev_index, counter))

    return index_sets


def remove_silence(input_audio: np.ndarray,
                   index_sets: list[tuple[int, int, int]]) -> np.ndarray:
    """
    Removes silent segments from an audio array based on the provided index sets.

    Args:
        input_audio (np.ndarray): Array of audio samples.
        index_sets (list[tuple[int, int, int]]): List of tuples representing silent segments.

    Returns:
        np.ndarray: Audio array with silent segments removed.
    """
    for silent_indices in index_sets:
        first_indx, last_idx = silent_indices[0], silent_indices[1]
        input_audio = np.delete(input_audio, range(first_indx, last_idx))
    return input_audio


def get_sequence_with_singing_indices(
        full_sequence: np.ndarray,
        chunk_length: int = 800) -> np.ndarray:
    """
    Identifies indices of sequences containing singing based on signal energy.

    Args:
        full_sequence (np.ndarray): Array of audio samples.
        chunk_length (int, optional): Length of chunks for energy computation. Defaults to 800.

    Returns:
        np.ndarray: Indices indicating the start and end of singing segments.
    """
    signal_magnitude = np.abs(full_sequence)

    chunks_energies = [
        np.mean(signal_magnitude[i:i + chunk_length])
        for i in range(0, len(signal_magnitude), chunk_length)
    ]

    threshold = np.max(chunks_energies) * 0.1
    chunks_energies = np.asarray(chunks_energies)
    chunks_energies[np.where(chunks_energies < threshold)] = 0
    onsets = np.zeros(len(chunks_energies))
    onsets[np.nonzero(chunks_energies)] = 1
    onsets = np.diff(onsets)

    start_ind = np.squeeze(np.where(onsets == 1))
    finish_ind = np.squeeze(np.where(onsets == -1))

    if finish_ind[0] < start_ind[0]:
        finish_ind = finish_ind[1:]

    if start_ind[-1] > finish_ind[-1]:
        start_ind = start_ind[:-1]

    indices_inici_final = np.insert(finish_ind, np.arange(len(start_ind)), start_ind)

    return np.squeeze((np.asarray(indices_inici_final) + 1) * chunk_length)
