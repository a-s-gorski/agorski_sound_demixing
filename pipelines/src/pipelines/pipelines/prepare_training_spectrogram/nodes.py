import logging
from typing import Dict, Tuple

import torch
from sklearn.model_selection import train_test_split

from pipelines.dataset.base_dataset import BaseDataset
from pipelines.types.prepare_spectrogram import SpectrogramConfig
from pipelines.utils.process_spectrogram import compute_spectrogram


def preproces_spectrogram_node(dataset: Dict[str,
                                             Tuple[torch.Tensor,
                                                   torch.Tensor]],
                               params: Dict[str,
                                            bool | int]) -> Tuple[Dict[str,
                                                                       Tuple[torch.Tensor,
                                                                             torch.Tensor]],
                                                                  Dict[str,
                                                                       float]]:
    """
    Preprocesses the spectrogram data based on the given parameters.

    Parameters:
        dataset (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): A dictionary containing the spectrogram dataset, where
            the keys 'train' and 'test' correspond to the training and test sets, respectively.
            Each value is a tuple containing two torch.Tensor objects representing the input spectrogram data and the
            target spectrogram data.
        params (Dict[str, bool | int]): A dictionary containing configuration parameters for computing the spectrogram.
            It should include the necessary settings to initialize a SpectrogramConfig object.

    Returns:
        Dict[str, Tuple[torch.Tensor, torch.Tensor]]: A dictionary containing the preprocessed spectrogram data.
            The keys 'train' and 'test' correspond to the preprocessed training and test sets, respectively.
            Each value is a tuple containing two torch.Tensor objects representing the processed input and target
            spectrogram data.

    Raises:
        Any exceptions or errors that can occur during the computation.
    """

    logger = logging.getLogger(__name__)

    config = SpectrogramConfig(**params)

    train_X, train_Y = dataset['train']
    test_X, test_Y = dataset['test']

    logger.info("Computing spectrograms.")

    processed_train_X, orginal_sizes = compute_spectrogram(
        train_X, config=config)

    processed_train_Y, _ = compute_spectrogram(train_Y, config=config)
    processed_test_X, _ = compute_spectrogram(test_X, config=config)
    processed_test_Y, _ = compute_spectrogram(test_Y, config=config)

    print(orginal_sizes)

    logger.info(
        f"Spectrograms computed successfully with orginal sizes: {orginal_sizes[0]} amd {orginal_sizes[1]}")

    data = {
        "train": (processed_train_X, processed_train_Y),
        "test": (processed_test_X, processed_test_Y),
    }

    original_shape_dict = {
        "width": orginal_sizes[0],
        "height": orginal_sizes[1],
    }

    return data, original_shape_dict


def spectrogram_split_node(dataset: Dict[str,
                                         Tuple[torch.Tensor,
                                               torch.Tensor]],
                           test_size: int,
                           random_state: int) -> Tuple[BaseDataset,
                                                       BaseDataset,
                                                       BaseDataset]:
    """
    Splits the spectrogram dataset into training, validation, and test sets.

    Parameters:
        dataset (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): A dictionary containing the spectrogram dataset, where
            the keys 'train' and 'test' correspond to the training and test sets, respectively.
            Each value is a tuple containing two torch.Tensor objects representing the input spectrogram data and the
            target spectrogram data.
        test_size (int): The number of samples to include in the test set during the split.
        random_state (int): The seed value used by the random number generator for reproducibility.

    Returns:
        Tuple[BaseDataset, BaseDataset, BaseDataset]: A tuple containing three BaseDataset objects representing the
            training, validation, and test datasets, respectively. Each BaseDataset object holds a subset of the
            spectrogram data with corresponding input and target data.

    Raises:
        Any exceptions or errors that can occur during the dataset split or BaseDataset initialization.
    """

    logger = logging.getLogger(__name__)

    train_X, train_Y = dataset['train']
    test_X, test_Y = dataset['test']

    train_X = train_X.numpy()
    train_Y = train_Y.numpy()

    logger.info("Spliting train spectrograms into train and val spectrograms.")

    train_X, val_X, train_Y, val_Y = train_test_split(
        train_X, train_Y, test_size=test_size, random_state=random_state)

    train_ds = BaseDataset(train_X, train_Y)
    val_ds = BaseDataset(val_X, val_Y)
    test_ds = BaseDataset(test_X, test_Y)

    return train_ds, val_ds, test_ds
