from pydantic import BaseModel
from typing import List


class DatasetConfig(BaseModel):
    sr: int = 44100
    channels: int = 1
    min_duration: float = 12.0
    max_duration: float = 640.0
    aug_shift: bool = True
    sample_length: int = 262144
    stems: List[str] = ['bass', 'drums', 'vocals', 'other']