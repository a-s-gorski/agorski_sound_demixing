from pipelines.dataset.base_dataset import BaseDataset
from pipelines.utils.train_signal_enriched import compute_enriched, signal_len_is_power_of_2, pad_signal_to_power_of_2, get_closest_larger_power_of_2, pad_dataset, remove_mono_dim_dataset, remove_mono_dim_signal, get_model, prepare_model_input, run_inference
from pipelines.types.training_waveform import TrainingWaveformConfig, WaveformModel, WaveformLoss
from pipelines.utils.training_waveform import get_model
from pipelines.utils.train_signal_enriched import get_model as get_generative_model
from pipelines.types.training_signal_enriched import TrainingGenerativeConfig, GenerativeModel
import pytest
import torch
import random
import string
from typing import List
from pipelines.models.gan import UnetGAN
from pipelines.models.vae import VAE


@pytest.mark.parametrize(
    "num_samples,num_sources,subseq_len,expected_num_samples,expected_num_sources,expected_sample_len",
    [
        (5, 4, 10000, 5, 4, 10000),
        (2, 2, 5000, 2, 2, 5000),
    ]
)
def test_compute_enriched(num_samples: int, num_sources: int, subseq_len: int, expected_num_samples: int, expected_num_sources: int, expected_sample_len: int):
    config = TrainingWaveformConfig(
        model=WaveformModel.CONVTASNET,
        loss=WaveformLoss.MSE,
        sources=[''.join(random.choice(string.ascii_letters)
                         for _ in range(10)) for _ in range(num_sources)],
        num_channels=1,
    )
    model = get_model(config=config)
    dataset = BaseDataset(
        X=torch.randn(num_samples, 1, subseq_len),
        Y=torch.randn(num_samples, num_sources, 1, subseq_len)
    )

    ds = compute_enriched(model, dataset)

    assert ds.X.shape == ds.Y.shape
    assert ds.X.shape[0] == expected_num_samples and ds.Y.shape[0] == expected_num_samples
    assert ds.X.shape[1] == expected_num_sources and ds.Y.shape[1] == expected_num_sources
    assert ds.X.shape[-1] == expected_sample_len and ds.Y.shape[-1] == expected_sample_len


@pytest.mark.parametrize("signal, expected_result", [
    (torch.tensor([1, 2, 3, 4]), True),
    (torch.tensor([1, 2, 3, 4, 5]), False),
    (torch.tensor([]), False)
])
def test_signal_len_is_power_of_2(signal, expected_result):
    assert signal_len_is_power_of_2(signal) == expected_result


@pytest.mark.parametrize("signal, expected_result", [
    (torch.tensor([1, 2, 3, 3]), 4),
    (torch.tensor([1, 2, 3, 4, 5]), 8),
    (torch.tensor([1, 2, 3, 4, 4]), 8),

])
def test_pad_signal_to_power_of_2(signal, expected_result):
    assert pad_signal_to_power_of_2(signal).shape[-1] == expected_result


@pytest.mark.parametrize("length, expected_result", [
    (1.5, 2),
    (3, 4),
    (7, 8),
])
def test_get_closest_larger_power_of_2(length, expected_result):
    assert get_closest_larger_power_of_2(length) == expected_result


@pytest.mark.parametrize("length, expected_result", [
    (30, 32),
    (5000, 8192),
    (10000, 16384),
])
def test_pad_dataset(length, expected_result):
    X = torch.randn(10, 1, length)
    Y = torch.randn(10, 4, 1, length)
    ds = BaseDataset(X, Y)
    padded_ds = pad_dataset(ds)   
@pytest.mark.parametrize("signal,num_sources,expected_shape", [
    (torch.randn(10, 4, 1, 10000), 4, [10, 4, 10000]),
    (torch.randn(10, 8, 1, 500), 8, [10, 8, 500]),
    (torch.randn(5, 12, 1, 1), 12, [5, 12, 1]),
])
def test_remove_mono_dim_signal(signal: torch.Tensor, num_sources: int, expected_shape: List[int]):

    assert (remove_mono_dim_signal(signal=signal, num_sources=num_sources)
            ).shape == torch.Size(expected_shape)


@pytest.mark.parametrize("num_samples,num_sources,num_channels,sample_len,expected_shape_x,expected_shape_y", [
    (10, 4, 1, 100, [10, 4, 100], [10, 4, 100]),
    (10, 8, 1, 500, [10, 8, 500], [10, 8, 500]),
    (10, 2, 1, 10000, [10, 2, 10000], [10, 2, 10000]),
])
def test_remove_mono_dim_dataset(num_samples: int, num_sources: int, num_channels: int, sample_len: int, expected_shape_x: List[int], expected_shape_y: List[int]):
    X = torch.randn(num_samples, num_sources, num_channels, sample_len)
    Y = torch.randn(num_samples, num_sources, num_channels, sample_len)
    ds = BaseDataset(X, Y)
    processed_ds = remove_mono_dim_dataset(ds, num_sources=num_sources)
    assert processed_ds.X.shape == torch.Size(
        expected_shape_x) and processed_ds.Y.shape == torch.Size(expected_shape_y)


def test_get_model():
    config = TrainingGenerativeConfig(
        model=GenerativeModel.GAN,
        loss=WaveformLoss.MSE,
        sources=[''.join(random.choice(string.ascii_letters)
                         for _ in range(10)) for _ in range(4)],
        input_channels=1,
        output_channels=1,
    )

    model = get_generative_model(config=config)
    assert isinstance(model, UnetGAN)

    config.model = GenerativeModel.VAE
    model = get_generative_model(config=config)
    assert isinstance(model, VAE)

@pytest.mark.parametrize("num_samples,num_sources,num_channels,sample_len,expected_shape_x,expected_shape_y",
    [
        (10, 4, 1, 1000, [10, 1, 4000], [10, 1, 4000])
    
    ],
)
def test_prepare_model_input(num_samples: int, num_sources: int, num_channels: int, sample_len: int, expected_shape_x: List[int], expected_shape_y: List[int]):
    config = TrainingGenerativeConfig(
        model=GenerativeModel.GAN,
        loss=WaveformLoss.MSE,
        sources=[''.join(random.choice(string.ascii_letters)
                         for _ in range(10)) for _ in range(num_sources)],
        input_channels=1,
        output_channels=1,
    )
    ds = BaseDataset(
        X=torch.randn(num_samples, num_sources, num_channels, sample_len),
        Y=torch.randn(num_samples, num_sources, num_channels, sample_len),
    )
    dataset, config, original_signal_len = prepare_model_input(config=config, dataset=ds)

    assert dataset.X.shape == torch.Size(expected_shape_x)
    assert dataset.Y.shape == torch.Size(expected_shape_y)



@pytest.mark.parametrize("model_type,signal_shape,expected_shape",
    [
        (GenerativeModel.VAE, [18, 4, 10000], [18, 4, 10000]),
        (GenerativeModel.GAN, [18, 4, 10000], [18, 4, 10000])
    ]
        
)
def test_run_inference(model_type: GenerativeModel,signal_shape: List[int],expected_shape: List[int]):
    config = TrainingGenerativeConfig(
        model=model_type,
        loss=WaveformLoss.MSE,
        sources=[''.join(random.choice(string.ascii_letters)
                         for _ in range(10)) for _ in range(signal_shape[1])],
        input_channels=4,
        output_channels=4,
        batch_size=2,
        num_workers=12,
        latent_dim= 64,
        layer_n= 128,
        hidden= 64,
        signal_len= 10000,
        learning_rate= 0.01,
        weight_decay= 0.00001,
        eps= 0.00000001,
        max_epochs= 3,
    )

    model = get_generative_model(config=config)
    test_dataset = BaseDataset(
        X=torch.randn(*signal_shape),
        Y=torch.randn(*signal_shape)
    )
    preds = run_inference(model, test_dataset)
    assert preds.shape == torch.Size(expected_shape)

if __name__ == "__main__":
    pytest.main()
