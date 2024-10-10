import pytest
from pipelines.utils.training_spectrogram import get_model, get_predictions, convert_spectrogram_to_signal
from pipelines.types.training_spectrogram import SpectrogramModel, SpectrogramLoss, TrainingSpectrogramConfig
import torch
from pipelines.models.unet import Unet
import random
import string
from typing import Dict, List

def test_get_model():
    config = TrainingSpectrogramConfig(
        model=SpectrogramModel.UNET,
        loss=SpectrogramLoss.MAE,
        input_channels=4,
        sources = ["bass.wav", "vocals.wav"]
    )
    model = get_model(config=config)
    assert isinstance(model, Unet)


@pytest.mark.parametrize("input_channels,num_sources,spectrograms,expected_shape",
        [
            (4, 2, torch.randn(2, 4, 256, 256), [2, 2, 1, 256, 256]),
            (8, 3, torch.randn(3, 8, 512, 512), [3, 3, 1, 512, 512])

        ]
)
def test_get_predictions(input_channels: int, num_sources: int, spectrograms: torch.randn, expected_shape: List[int]):
    config = TrainingSpectrogramConfig(
        model=SpectrogramModel.UNET,
        loss=SpectrogramLoss.MAE,
        input_channels=input_channels,
        sources = ["".join([random.choice(string.ascii_lowercase) for _ in range(10)]) for _ in range(num_sources)]
    )
    model = get_model(config=config)
    preds = get_predictions(model, spectrograms=spectrograms)
    assert preds.shape == torch.Size(expected_shape)


@pytest.mark.parametrize("spectrograms,orginal_shapes,expected_shapes", 
    [
        (torch.randn(18, 4, 1, 256, 256), {"height": 51, "width": 201}, [18, 4, 1, 10000])
                             
    ])
def test_convert_spectrogram_to_signal(spectrograms: torch.Tensor, orginal_shapes: Dict[str, int], expected_shapes: List[int]):
    waveform = convert_spectrogram_to_signal(spectrograms, orginal_shapes=orginal_shapes)
    
    assert waveform.shape == torch.Size(expected_shapes)

if __name__ == "__main__":
    pytest.main()
