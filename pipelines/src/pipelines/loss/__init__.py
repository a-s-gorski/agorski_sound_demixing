import math
from typing import Callable

import torch
import torch.nn as nn
from torchlibrosa.stft import STFT

from pipelines.loss.loss import L1_Wav_L1_CompressedSp, L1_Wav_L1_Sp, l1_wav

# from bytesep.models.pytorch_modules import Base
from pipelines.utils.pytorch_modules import Base


def get_loss_function(loss_type: str) -> Callable:
    r"""Get loss function.

    Args:
        loss_type: str

    Returns:
        loss function: Callable
    """

    if loss_type == "l1_wav":
        return l1_wav

    elif loss_type == "l1_wav_l1_sp":
        return L1_Wav_L1_Sp()

    elif loss_type == "l1_wav_l1_compressed_sp":
        return L1_Wav_L1_CompressedSp()

    else:
        raise NotImplementedError
