import os
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from numpy.typing import NDArray
from torch.nn import functional as F
from tqdm import tqdm

from pipelines.types.prepare_signal import PrepareSignalConfig
from pipelines.types.signal import SignalProcessingConfig, SoundType


def extract_signals(
        path: str, config: SignalProcessingConfig) -> Tuple[NDArray, NDArray]:
    """
    Uses librosa to read data from musdb18 dataset.

    Parameters:
        path(str): Path to the musdb18 download location
        config(SignalProcessingConfig): Config for loading dataset

    Returns:
        numpy.NDArray: original signal with shape (num_samples, channels, subsequence_shape)
        numpy.NDArray: seperated sources (drums, bass, other, vocals) with shape (num_samples, num_sources = 4, channels, subsequence_shape)
    """
    X, Y = [], []
    # TODO - remember to replace with larger value later
    for track in tqdm(os.listdir(path)[:50]):
        if config.input_source not in os.listdir(os.path.join(path, track)):
            continue
        instrument_dict = {
            source: None for source in config.sources + [config.input_source, ]}
        for instrument in os.listdir(os.path.join(path, track)):
            signal, sr = librosa.load(os.path.join(path, track, instrument), sr=config.sr,
                                      mono=config.signal_type == SoundType.MONO, res_type="kaiser_fast", offset=0.0)
            if config.signal_type == SoundType.MONO:
                signal = signal.reshape(1, -1)
            signal = signal[:, :config.max_signal_size]
            signal = signal[:, :signal.shape[1] -
                            signal.shape[1] %
                            config.subsequence_len]
            signal = signal.reshape(-1,
                                    signal.shape[0],
                                    config.subsequence_len)
            instrument_dict[instrument] = signal
        desired_shape = instrument_dict[config.input_source].shape
        for instrument, signal in instrument_dict.items():
            if signal is None:
                instrument_dict[instrument] = np.zeros(desired_shape)
        x = instrument_dict[config.input_source]
        y = np.concatenate(
            [
                np.expand_dims(
                    signal,
                    axis=1) for instrument,
                signal in instrument_dict.items() if instrument != config.input_source],
            axis=1)
        X.append(x)
        Y.append(y)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y


def get_effects_list(config: PrepareSignalConfig) -> List[List[str]]:
    """
    Generates a list of effect commands to be applied to audio signals based on the provided configuration.

    Parameters:
        config (PrepareSignalConfig): An instance of the PrepareSignalConfig class containing configuration settings.

    Returns:
        List[List[str]]: A list of effect commands in the format of lists of strings.
            Each inner list represents an effect command with its arguments.
    """
    effects = []
    if config.resample:
        effects.append(["rate", str(config.resample_freq)])
    if config.highpass:
        effects.append(["highpass", "-1", str(config.highpass_freq)])
    if config.lowpass:
        effects.append(["lowpass", "-1", str(config.lowpass_freq)])
    if config.speed:
        effects.append(["speed", str(config.speed_ratio)])
    if config.reverb:
        effects.append(["reverb", "-w"])
    if config.channels:
        effects.append(["channels", str(config.channels_num)])
    if config.normalize:
        effects.append(["norm"])
    return effects


def apply_signal(
        signal: torch.Tensor,
        signal_rate: int,
        subseq_len: int,
        effects: List[List[str]]) -> torch.Tensor:
    """
    Applies audio effects to a given audio signal tensor and returns the processed signal.

    Parameters:
        signal (torch.Tensor): The input audio signal as a torch tensor.
        signal_rate (int): The sample rate of the input audio signal.
        subseq_len (int): The desired length of the output audio signal after applying effects.
        effects (List[str]): A list of effect commands in the format of strings.

    Returns:
        torch.Tensor: The processed audio signal tensor after applying the effects.
    """
    if effects:
        signal = torchaudio.sox_effects.apply_effects_tensor(
            signal, signal_rate, effects)[0]
    processed_signal = signal[:, :subseq_len]
    processed_signal = F.pad(
        processed_signal, (0, subseq_len - processed_signal.shape[1]))
    return processed_signal


def transform_signals(X_signal: NDArray,
                      Y_signal: NDArray,
                      config: PrepareSignalConfig) -> Tuple[torch.Tensor,
                                                            torch.Tensor]:
    """
    Transforms input and output audio signals using the provided configuration.

    Parameters:
        X_signal (NDArray): The input audio signals as a NumPy array.
        Y_signal (NDArray): The output audio signals as a NumPy array.
        config (PrepareSignalConfig): An instance of the PrepareSignalConfig class containing configuration settings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two torch tensors.
            - The first tensor represents the transformed input audio signals (X).
            - The second tensor represents the transformed output audio signals (Y).
    """
    X_signal = torch.from_numpy(X_signal).to(torch.float32)
    Y_signal = torch.from_numpy(Y_signal).to(torch.float32)
    X = []
    Y = []
    effects = get_effects_list(config)
    sr = config.sample_freq
    for input_signal, output_signals in tqdm(
            zip(X_signal, Y_signal), total=min(len(X_signal), len(Y_signal))):
        processed_input = apply_signal(
            input_signal, sr, config.subseq_len, effects)
        processed_output = list(map(lambda signal: apply_signal(
            signal, sr, config.subseq_len, effects), output_signals))
        processed_output = torch.cat(list(
            map(lambda signal: torch.unsqueeze(signal, axis=0), processed_output)), axis=0)
        X.append(processed_input.unsqueeze(0))
        Y.append(processed_output.unsqueeze(0))
    X = torch.cat(X, axis=0)
    Y = torch.cat(Y, axis=0)
    return X, Y
