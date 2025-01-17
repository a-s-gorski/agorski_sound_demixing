import torchaudio
from typing import Union, List, Tuple
import torch
from pathlib import Path


def load_audio_tracks(paths: List[Union[str, Path]], sample_rate: int) -> Tuple[torch.Tensor, ...]:
    signals, sample_rates = zip(*[torchaudio.load(path) for path in paths])
    for sr in sample_rates:
        assert sr == sample_rate, f"sample rate {sr} is different from target sample rate {sample_rate}"
    return tuple(signals)


def assert_is_audio(*signal: torch.Tensor):
    for s in signal:
        assert len(s.shape) == 2
        assert s.shape[0] == 1 or s.shape[0] == 2


def is_silent(signal: torch.Tensor, silence_threshold: float = 1.5e-5) -> bool:
    assert_is_audio(signal)
    num_samples = signal.shape[-1]
    return torch.linalg.norm(signal) / num_samples < silence_threshold


def is_multi_source(signal: torch.Tensor, silence_threshold: float = 1.5e-5) -> bool:
    num_silent_signals = 0
    for source in signal:
        if is_silent(source.unsqueeze(0), silence_threshold):
            num_silent_signals += 1
        if num_silent_signals > 2:
            return False
    return True  
    

def get_nonsilent_and_multi_instr_chunks(
    separated_track: Tuple[torch.Tensor],
    max_chunk_size: int,
    min_chunk_size: int,
    silence_threshold: Union[float,None],
    keep_only_multisource: bool ,
):
    for source in separated_track:
        assert_is_audio(source)
    
    separated_track = torch.cat(separated_track)
    _, num_samples = separated_track.shape
    num_chunks = num_samples // max_chunk_size + int(num_samples % max_chunk_size != 0)

    available_chunks = []
    for i in range(num_chunks):
        chunk = separated_track[:, i * max_chunk_size : (i + 1) * max_chunk_size]
        _, chunk_samples = chunk.shape

        # Remove if silent
        if silence_threshold is not None and is_silent(chunk.sum(0, keepdims=True), silence_threshold):
            continue
        
        # Remove if it contains only one source
        if keep_only_multisource and not is_multi_source(chunk):
            continue
        
        # Remove if it contains less than the minimum chunk size
        if chunk_samples < min_chunk_size:
            continue
        
        available_chunks.append(i)
    return available_chunks