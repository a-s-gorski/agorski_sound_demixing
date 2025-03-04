import numpy as np
from pipelines.gan.config import TrainingGANConfig

def _any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return np.any(np.all(np.sum(
        sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))

def audio_generator(audio_data, config: TrainingGANConfig):
    audio_len = len(audio_data)
    n_iters = audio_len // config.window_length
    for i in range(n_iters+1):
        start_idx = i * config.window_length
        end_idx = start_idx  + config.window_length
        result = np.zeros(config.window_length)
        audio_size = audio_data[start_idx:end_idx].shape[0]
        result[:audio_size] = audio_data[start_idx:end_idx]
        yield result