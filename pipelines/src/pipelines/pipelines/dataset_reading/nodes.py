import logging
import os
from typing import Dict, List

from numpy.typing import NDArray

from pipelines.types.signal import SignalProcessingConfig
from pipelines.utils.process_signals import extract_signals


def process_dataset_signals(paths: Dict[str,
                                        str],
                            params: Dict[str,
                                         int | str | List[str]]) -> Dict[str,
                                                                         Dict[str,
                                                                              int | NDArray]]:
    """
    Process MUSDB18 dataset signals using the provided parameters.
    The directory structure should follow https://dagshub.com/kinkusuma/musdb18-dataset
    aka
    train:
        - song1
            - your_target_name.wav
            - your_source_name1.wav
            - your_source_name2.wav
            ...
    test:
        - songn
            - your_target_name.wav
            - your_source_name1.wav
            - your_source_name2.wav
            ...

    Args:
        paths (Dict[str, str]): A dictionary containing the input and output paths for processing.
            - 'input_path' (str): The path to the input dataset.

        params (Dict[str, Union[int, str, List[str]]]): A dictionary containing parameters for signal processing.
            - 'signal_type' (str): The type of sound being processed ('mono' or 'stereo').
            - 'max_signal_size' (int): The maximum size of the signal.
            - 'subsequence_len' (int): The length of subsequences used in processing.
            - 'sources' (List[str]): List of sources used for processing.
            - 'input_source' (str): The input source for processing.
            - 'sr' (int): The sample rate of the signal.

    Returns:
        Dict[str, Dict[str, Union[int, np.ndarray]]]: A dictionary containing the processed signals for the train and test sets.
            - 'train' (Dict[str, Union[int, np.ndarray]]): A dictionary containing the processed signals for the train set.
                - 'signal_type' (int): The type of sound being processed ('mono' or 'stereo').
                - 'signal_data' (np.ndarray): A 2D NumPy array representing the processed signals of the train set.
            - 'test' (Dict[str, Union[int, np.ndarray]]): A dictionary containing the processed signals for the test set.
                - 'signal_type' (int): The type of sound being processed ('mono' or 'stereo').
                - 'signal_data' (np.ndarray): A 2D NumPy array representing the processed signals of the test set.
    """

    logger = logging.getLogger(__name__)

    input_path = paths["input_path"]

    logging.info("Loading configuration file.")
    data_processing_config = SignalProcessingConfig(**params)

    train_path, test_path = os.path.join(
        input_path, 'train'), os.path.join(
        input_path, 'test')

    logger.info("Loading training signals.")
    train = extract_signals(train_path, data_processing_config)

    logger.info("Loading test signals.")
    test = extract_signals(test_path, data_processing_config)

    return {'train': train, "test": test}
