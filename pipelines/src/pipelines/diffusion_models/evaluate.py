from pathlib import Path
from typing import Union, Sequence
import pandas as pd
from collections import defaultdict
import math

from pipelines.diffusion_models.utils import get_full_tracks, load_and_resample_track, is_silent
from pipelines.diffusion_models.metrics import sisnr

def evaluate_separations(
    separation_path: Union[str, Path],
    dataset_path: Union[str, Path],
    separation_sr: int,
    filter_single_source: bool = True,
    stems: Sequence[str] = ("bass","drums","mixture","other"),
    eps: float = 1e-8,
    chunk_duration: float = 4.0, 
    overlap_duration: float = 2.0
) -> pd.DataFrame:

    separation_path = Path(separation_path)
    dataset_path = Path(dataset_path)
    
    df_entries = defaultdict(list)
    for track, separated_track in get_full_tracks(separation_path, separation_sr, stems):
        
        # load and resample track
        original_track = load_and_resample_track(dataset_path/track, stems, 22050)

        # Adjust for changes in length
        for s in stems:
            max_length = separated_track[s].shape[-1]
            original_track[s] = original_track[s][:,:max_length]
        
        # Compute mixture
        mixture = sum([owav for owav in original_track.values()])

        chunk_samples = int(chunk_duration * separation_sr)
        overlap_samples = int(overlap_duration * separation_sr)

        # Calculate the step size between consecutive sub-chunks
        step_size = chunk_samples - overlap_samples

        # Determine the number of evaluation chunks based on step_size
        num_eval_chunks = math.ceil((mixture.shape[-1] - overlap_samples) / step_size)
            
        for i in range(num_eval_chunks):
            start_sample = i * step_size
            end_sample = start_sample + chunk_samples
            
            # Determine number of active signals in sub-chunk
            num_active_signals = 0
            for k in separated_track:
                o = original_track[k][:,start_sample:end_sample]
                if not is_silent(o):
                    num_active_signals += 1
            
            # Skip sub-chunk if necessary
            if filter_single_source and num_active_signals <= 1:
                continue

            # Compute SI-SNRi for each stem
            for k in separated_track:
                o = original_track[k][:,start_sample:end_sample]
                s = separated_track[k][:,start_sample:end_sample]
                m = mixture[:,start_sample:end_sample]
                df_entries[k].append((sisnr(s, o, eps) - sisnr(m, o, eps))[0].item())
            
            # Add chunk and sub-chunk info to dataframe entry
            df_entries["start_sample"].append(start_sample)
            df_entries["end_sample"].append(end_sample)

    # Create and return dataframe
    return pd.DataFrame(df_entries)
