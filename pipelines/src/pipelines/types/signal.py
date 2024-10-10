from enum import Enum
from typing import List

from pydantic import BaseModel


class SoundType(Enum):
    """
    Enum class representing the type of sound.

    Attributes:
        MONO (str): Mono sound type.
        STEREO (str): Stereo sound type.
    """
    MONO = "mono"
    STEREO = "stereo"


class SignalProcessingConfig(BaseModel):
    """
    A configuration model for signal processing.

    Attributes:
        signal_type (SoundType): The type of sound being processed (mono or stereo).
        max_signal_size (int): The maximum size of the signal.
        subsequence_len (int): The length of subsequences used in processing.
        sources (List[str]): List of sources used for processing.
        input_source (str): The input source for processing.
        sr (int): The sample rate of the signal.
    """
    signal_type: SoundType
    max_signal_size: int
    subsequence_len: int
    sources: List[str]
    input_source: str
    sr: int
