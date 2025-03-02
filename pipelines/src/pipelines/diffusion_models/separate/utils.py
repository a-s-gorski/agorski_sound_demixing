import torch
from typing import Tuple, Union, Callable, Optional, Mapping
from tqdm import tqdm
from math import sqrt
import torchaudio
from pathlib import Path

from pipelines.diffusion_models.utils import is_silent, assert_is_audio, is_multi_source
from pipelines.diffusion_models.differential import differential_with_dirac

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

@torch.no_grad()
def separate_mixture(
    mixture: torch.Tensor, 
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    noises: Optional[torch.Tensor],
    differential_fn: Callable = differential_with_dirac,
    s_churn: float = 0.0, # > 0 to add randomness
    num_resamples: int = 1,
    use_tqdm: bool = False,
):      
    # Set initial noise
    x = sigmas[0] * noises # [batch_size, num-sources, sample-length]
    
    vis_wrapper  = tqdm.tqdm if use_tqdm else lambda x:x 
    for i in vis_wrapper(range(len(sigmas) - 1)):
        sigma, sigma_next = sigmas[i], sigmas[i+1]

        for r in range(num_resamples):
            # Inject randomness
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
            sigma_hat = sigma * (gamma + 1)
            x = x + torch.randn_like(x) * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            # Compute conditioned derivative
            d = differential_fn(mixture=mixture, x=x, sigma=sigma_hat, denoise_fn=denoise_fn)

            # Update integral
            x = x + d * (sigma_next - sigma_hat)

            # Renoise if not last resample step
            if r < num_resamples - 1:
                x = x + sqrt(sigma ** 2 - sigma_next ** 2) * torch.randn_like(x)
    
    return x.cpu().detach()

def save_separation(
    separated_tracks: Mapping[str, torch.Tensor],
    sample_rate: int,
    chunk_path: Path,
):    
    for stem, separated_track in separated_tracks.items():
        assert_is_audio(separated_track)
        torchaudio.save(chunk_path / f"{stem}.wav", separated_track.cpu(), sample_rate=sample_rate)
