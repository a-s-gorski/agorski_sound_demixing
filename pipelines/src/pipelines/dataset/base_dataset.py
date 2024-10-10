from typing import Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]
