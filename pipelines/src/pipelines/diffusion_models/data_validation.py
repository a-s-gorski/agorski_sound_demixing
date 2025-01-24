import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DatamoduleWithValidation(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that handles training and validation datasets.
    
    Attributes:
        data_train: The dataset used for training.
        data_val: The dataset used for validation.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses used for data loading.
        pin_memory: If True, the data loader will copy tensors into CUDA pinned memory.
    """
    
    def __init__(
        self,
        train_dataset,
        val_dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        """
        Initializes the data module with training and validation datasets.
        
        Args:
            train_dataset: A dataset instance used for training.
            val_dataset: A dataset instance used for validation.
            batch_size: Number of samples per batch.
            num_workers: Number of worker threads for data loading.
            pin_memory: Whether to use pinned memory for faster GPU transfer.
        """
        super().__init__()
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.
        
        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.
        
        Returns:
            DataLoader: A PyTorch DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
