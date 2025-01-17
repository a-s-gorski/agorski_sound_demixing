from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pipelines.diffusion_models.data.dataset import MultiSourceDataset

class DatamoduleWithValidation(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: MultiSourceDataset,
        val_dataset: MultiSourceDataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )