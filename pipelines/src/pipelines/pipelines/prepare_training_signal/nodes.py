import logging
from typing import Dict, Tuple

import torch
from sklearn.model_selection import train_test_split

from pipelines.dataset.base_dataset import BaseDataset
from pipelines.types.prepare_signal import PrepareSignalConfig
from pipelines.utils.process_signals import transform_signals


def signal_preprocessing_node(dataset: Dict[str,
                                            Tuple[torch.Tensor,
                                                  torch.Tensor]],
                              params: Dict[str,
                                           str | float]) -> Dict[str,
                                                                 Tuple[torch.Tensor,
                                                                       torch.Tensor]]:
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
        dataset (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): A dictionary containing the dataset with keys 'train' and 'test'.
            The values associated with these keys are tuples of torch Tensors representing the input signals and their respective labels.
        test_size (float): The proportion of the dataset to include in the test split (0.0 to 1.0).
        random_state (int): The seed used by the random number generator for reproducibility.

    Returns:
        Tuple[BaseDataset, BaseDataset, BaseDataset]: A tuple containing three BaseDataset objects representing the training, validation, and test sets.
            The BaseDataset class is a custom dataset class that holds torch Tensors of input signals and labels.
    """
    logger = logging.getLogger(__name__)

    train_X, train_Y = dataset['train']
    test_X, test_Y = dataset['test']
    config = PrepareSignalConfig(**params)

    logger.info("Processing train signal.")
    processed_train = transform_signals(train_X, train_Y, config)

    logger.info("Processing test signal.")
    processed_test = transform_signals(test_X, test_Y, config)

    data = {
        'train': processed_train,
        'test': processed_test,
    }

    return data


def signal_split_node(dataset: Dict[str,
                                    Tuple[torch.Tensor,
                                          torch.Tensor]],
                      test_size: float,
                      random_state: int) -> Tuple[BaseDataset,
                                                  BaseDataset,
                                                  BaseDataset]:
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
        dataset (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): A dictionary containing the dataset with keys 'train' and 'test'.
            The values associated with these keys are tuples of torch Tensors representing the input signals and their respective labels.
        test_size (float): The proportion of the dataset to include in the test split (0.0 to 1.0).
        random_state (int): The seed used by the random number generator for reproducibility.

    Returns:
        Tuple[BaseDataset, BaseDataset, BaseDataset]: A tuple containing three BaseDataset objects representing the training, validation, and test sets.
            The BaseDataset class is a custom dataset class that holds torch Tensors of input signals and labels.
    """

    logger = logging.getLogger(__name__)
    logger.info("Splitting train signal into train and val.")

    train_X, train_Y = dataset['train']
    test_X, test_Y = dataset['test']
    train_X, train_Y = train_X.numpy(), train_Y.numpy()
    train_X, val_X, train_Y, val_Y = train_test_split(
        train_X, train_Y, test_size=test_size, random_state=random_state)
    train_X = torch.from_numpy(train_X)
    train_Y = torch.from_numpy(train_Y)
    val_X = torch.from_numpy(val_X)
    val_Y = torch.from_numpy(val_Y)
    train_ds = BaseDataset(train_X, train_Y)
    val_ds = BaseDataset(val_X, val_Y)
    test_ds = BaseDataset(test_X, test_Y)

    return train_ds, val_ds, test_ds
