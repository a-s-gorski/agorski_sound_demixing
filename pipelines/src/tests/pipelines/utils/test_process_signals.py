import pytest
import tempfile
import os
import numpy as np
import soundfile as sf
from typing import List, Callable
from pipelines.utils.process_signals import extract_signals, get_effects_list, apply_signal, transform_signals
from pipelines.types.signal import SignalProcessingConfig, SoundType
from pipelines.types.prepare_signal import PrepareSignalConfig
import shutil
import torch


@pytest.fixture(scope="module")
def setup_mock_filesystem() -> str:
    temp_dir = os.path.join(tempfile.gettempdir(), "samples")
    os.makedirs(temp_dir, exist_ok=True)
    
    track_names = ["song1", "song2", "song3"]
    track_files = [["mixture.wav", "vocals.wav", "bass.wav"], ["bass.wav", "bass.wav", "vocals.wav"], ["mixture.wav", "bass.wav", "vocals.wav"]]
    output_filepaths = []

    for directory, sources_list in zip(track_names, track_files):
        os.makedirs(os.path.join(temp_dir, directory), exist_ok=True)
        output_filepaths.append(os.path.join(temp_dir, directory))
        for source in sources_list:
            file_path = os.path.join(temp_dir, directory, source)
            data = np.random.randn(10000, 1)
            sf.write(file_path, data, 22500, subtype='PCM_24')
            
    
    yield temp_dir

    shutil.rmtree(temp_dir)

@pytest.fixture()
def processing_config() -> SignalProcessingConfig:
    config = SignalProcessingConfig(
        signal_type=SoundType.MONO,
        max_signal_size=100000,
        subsequence_len=500,
        sources=["bass.wav", "vocals.wav"],
        input_source="mixture.wav",
        sr=22500
    )
    return config
    

# @pytest.fixture()
# def mock_filesystem(mocker, setup_mock_filesystem) -> None:
#     mocker.patch("os.listdir", return_value=setup_mock_filesystem)


def test_extract_signals(setup_mock_filesystem: Callable, processing_config):

    X, Y = extract_signals(setup_mock_filesystem, processing_config)

    # Test case 1 - check subsequence_len
    assert X.shape[-1] == 500 and Y.shape[-1] == 500

    # Test case 2 - check signal type
    assert X.shape[-2] == 1 and Y.shape[-2] == 1

    # Test case 3 - check number of sources    
    assert Y.shape[-3] == 2

    # Test if the number of samples for x and y is the same
    assert X.shape[0] == Y.shape[0]

    # Check if the total length of all subsequences is not greater than maximal signal length
    assert X.shape[0]*X.shape[1]*X.shape[2] <= 100000

@pytest.mark.parametrize(
    "config, expected_effects",
    [
        (
            PrepareSignalConfig(
                sample_freq=44100,
                normalize=False,
                highpass=False,
                highpass_freq=0,
                lowpass=False,
                lowpass_freq=0,
                resample=False,
                resample_freq=0,
                speed=False,
                speed_ratio=0,
                reverb=False,
                channels=False,
                channels_num=0,
                subseq_len=0,
            ),
            [],
        ),
        (
            PrepareSignalConfig(
                sample_freq=44100,
                normalize=True,
                highpass=True,
                highpass_freq=100,
                lowpass=True,
                lowpass_freq=5000,
                resample=True,
                resample_freq=22050,
                speed=True,
                speed_ratio=0.8,
                reverb=True,
                channels=True,
                channels_num=2,
                subseq_len=100,
            ),
            [
                ["rate", "22050"],
                ["highpass", "-1", "100"],
                ["lowpass", "-1", "5000"],
                ["speed", "0.8"],
                ["reverb", "-w"],
                ["channels", "2"],
                ["norm"],
            ],
        ),
        (
            PrepareSignalConfig(
                sample_freq=44100,
                normalize=False,
                highpass=False,
                highpass_freq=0,
                lowpass=False,
                lowpass_freq=0,
                resample=True,
                resample_freq=22050,
                speed=False,
                speed_ratio=0,
                reverb=False,
                channels=False,
                channels_num=0,
                subseq_len=0,
            ),
            [["rate", "22050"]],
        ),
        (
            PrepareSignalConfig(
                sample_freq=44100,
                normalize=True,
                highpass=True,
                highpass_freq=100,
                lowpass=False,
                lowpass_freq=0,
                resample=False,
                resample_freq=0,
                speed=False,
                speed_ratio=0,
                reverb=False,
                channels=False,
                channels_num=0,
                subseq_len=0,
            ),
            [["highpass", "-1", "100"], ["norm"]],
        ),
    ],
)
def test_get_effects_list(config, expected_effects):
    assert get_effects_list(config) == expected_effects



def test_apply_signal():
    signal = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    signal_rate = 44100
    subseq_len = 2
    effects = [["speed", "1.5"], ]
    
    processed_signal = apply_signal(signal, signal_rate, subseq_len, effects)

    assert processed_signal.shape == (2, subseq_len)
    # You can add more specific assertions based on the effects you're applying.
    # For example, check if the speed effect was applied correctly.


test_configs = [
    PrepareSignalConfig(
        sample_freq=44100,
        normalize=True,
        highpass=True,
        highpass_freq=100,
        lowpass=False,
        lowpass_freq=0,
        resample=False,
        resample_freq=0,
        speed=False,
        speed_ratio=0,
        reverb=False,
        channels=False,
        channels_num=1,
        subseq_len=1000,
    ),
    PrepareSignalConfig(
        sample_freq=22050,
        normalize=True,
        highpass=False,
        highpass_freq=0,
        lowpass=True,
        lowpass_freq=8000,
        resample=True,
        resample_freq=16000,
        speed=True,
        speed_ratio=1.5,
        reverb=True,
        channels=True,
        channels_num=2,
        subseq_len=500,
    ),
]

@pytest.mark.parametrize("config", test_configs)
def test_transform_signals(config):
    X_signal = np.random.randn(10, 1, 10000)
    Y_signal = np.random.rand(10, 4, 1, 10000) 
    X, Y = transform_signals(X_signal, Y_signal, config)

    assert X.shape == (10, config.channels_num, config.subseq_len)
    assert Y.shape == (10, 4, config.channels_num, config.subseq_len)

if __name__ == "__main__":
    pytest.main()
