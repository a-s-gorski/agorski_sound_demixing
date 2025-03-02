from tqdm import tqdm
from pathlib import Path
import json
from collections import defaultdict
from typing import Union, Sequence, Tuple, Mapping, List, Any
import torch
import torchaudio
from torchaudio.functional import resample

def stringify(obj:Union[Mapping,List,Tuple, Any]):
    if isinstance(obj, Mapping):
        return {k:stringify(v) for k,v in obj.items()}
    elif isinstance(obj, (List,Tuple)):
        return [stringify(v) for v in obj]
    else:
        return str(obj)

def load_chunks(chunk_folder: Path, stems: Sequence[str]) -> Tuple[Mapping[str, torch.Tensor], int]:
    separated_tracks_and_rate = {s: torchaudio.load(chunk_folder / f"{s}.wav") for s in stems}
    separated_tracks = {k:t for k, (t,_) in separated_tracks_and_rate.items()}
    sample_rates_sep = [s for (_,s) in separated_tracks_and_rate.values()]

    assert len({*sample_rates_sep}) == 1, print(sample_rates_sep)
    sr = sample_rates_sep[0]

    return separated_tracks, sr


def get_full_tracks(
    separation_path: Union[str, Path],
    expected_sample_rate: int = 22050,
    stems: Sequence[str] = ("bass","drums","mixture","other"),
):
    separation_folder = Path(separation_path)
    assert separation_folder.exists(), separation_folder
    assert (separation_folder / "chunk_data.json").exists(), separation_folder

    with open(separation_folder / "chunk_data.json") as f:
        chunk_data = json.load(f)

    track_to_chunks = defaultdict(list)
    for chunk_data in chunk_data:
        track = chunk_data["track"]
        chunk_idx = chunk_data["chunk_index"]
        start_sample = chunk_data["start_chunk_sample"]
        #track_sample_rate = chunk_data["sample_rate"]
        track_to_chunks[track].append( (start_sample, chunk_idx) )

    # Reorder chunks into ascending order
    for track, chunks in tqdm(track_to_chunks.items()):
        sorted_chunks = sorted(chunks)
        
        # Merge separations
        separated_wavs = {s: [] for s in stems}
        for _, chunk_idx in sorted_chunks:
            chunk_folder = separation_folder / str(chunk_idx)
            
            separated_chunks, sr = load_chunks(chunk_folder, stems)
            assert sr == expected_sample_rate, f"{sr} different from expected sample-rate {expected_sample_rate}"
            
            # convert start sample to the chunk sample-rate
            #start_sample = start_sample * sr // track_sample_rate

            for s in separated_chunks:
                separated_wavs[s].append(separated_chunks[s])

        for s in stems:
            separated_wavs[s] = torch.cat(separated_wavs[s], dim=-1)
            
        yield track, separated_wavs
        
def load_and_resample_track(track_path: Union[str,Path], stems: Sequence[str], resample_sr: int) -> Mapping[str, torch.Tensor]:
    track_path = Path(track_path)
    
    def _load_and_resample_stem(stem: Path):
        stem_path = track_path/f"{stem}.wav"
        if stem_path.exists():
            wav, sr = torchaudio.load(stem_path)
            return resample(wav, sr, resample_sr)
        else:
            return None
    
    # Load and resample track stems
    stem_to_track = {s:_load_and_resample_stem(s) for s in stems}
    
    # Assert it contains at least a single source
    assert set(stem_to_track.values()) != {None}
    
    # Get sources dimensionality
    shapes = {wav.shape for s, wav in stem_to_track.items() if wav is not None}
    num_channels = {channels for (channels,length) in shapes}
    sample_lengths = {length for (channels,length) in shapes} 
    
    # Assert the existing sources have same dimensionality (up to certaian threshold)
    assert len(num_channels) == 1, f"{num_channels}"
    num_channels, = num_channels
    assert max(sample_lengths) - min(sample_lengths) <= 0.1 * resample_sr, f"{(max(sample_lengths) - min(sample_lengths))/resample_sr}"
    
    for s, wav in stem_to_track.items():
        # Initialize missing sources to zero
        if wav is None:
            stem_to_track[s] = torch.zeros(size=(num_channels, min(sample_lengths)) )
        
        # Crop sources
        else:
            stem_to_track[s] = stem_to_track[s][:,:min(sample_lengths)]
    
    return stem_to_track

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