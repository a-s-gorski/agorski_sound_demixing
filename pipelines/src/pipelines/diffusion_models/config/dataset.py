from typing import List

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    sr: int = 44100
    sample_length: int = 262144
    channels: int = 1
    min_duration: float = 12.0
    max_duration: float = 640.0
    aug_shift: bool = True
    sample_length: int = 262144
    stems: List[str] = ['bass', 'drums', 'vocals', 'other']
