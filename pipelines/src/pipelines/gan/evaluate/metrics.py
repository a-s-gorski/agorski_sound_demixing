import torch
from tqdm import tqdm
import numpy as np
import mir_eval
import gc
from typing import Dict

from pipelines.gan.data.loader import LMDBWavLoader
from pipelines.gan.evaluate.utils import audio_generator
from pipelines.gan.model.cycle_gan import CycleGan
from pipelines.gan.config import TrainingGANConfig

def calculate_metrics(test_dataset: LMDBWavLoader, gan_model: CycleGan, config: TrainingGANConfig) -> Dict:
    test_data_len = len(test_dataset) 
    with torch.no_grad():
        # to get 1 min output
        reconstructed_signals = []
        mixed_signals = []
        original_signals = []
        out_sdr = []
        out_sir = []
        out_sar = []
        for i in tqdm(range(test_data_len)):
            #sample_rate = 20480
            val_data  = test_dataset[i] 
            mixed_signal = []
            wav_iter = list(audio_generator(val_data[1], config=config))
            subsample_n_samples = len(wav_iter)
            for j in range(subsample_n_samples):
                mixed_signal.append(wav_iter[j])
            mixed_signal = torch.squeeze(torch.from_numpy(np.stack(mixed_signal,axis=0)).float()).to("cuda")
            mixed_signal = torch.unsqueeze(mixed_signal, dim=1)
            reconstructed = gan_model.generator(mixed_signal) 
            single_source = val_data[0]
            
            n_items_per_eval = 3
            data_len = (single_source.shape[-1]//n_items_per_eval) * n_items_per_eval
            reconstructed = torch.squeeze(reconstructed).detach().cpu().numpy().reshape(1,-1)[:,:data_len]

            reconstructed = reconstructed[:data_len].reshape( -1)
            single_source = single_source[:data_len].reshape( -1) 
            mixed_source = val_data[1][:data_len].reshape( -1)
            clean_inference = mixed_source - single_source
            predicted_inference = mixed_source - reconstructed
            #single_source = librosa.resample(single_source, sample_rate, window_length) 
            #reconstructed = librosa.resample(reconstructed, sample_rate, window_length)
            reference_music =single_source#, mixed_signal[:data_len].reshape(n_items_per_eval, -1) - single_source[:data_len].reshape(n_items_per_eval, -1)])
            estimates_music = reconstructed#, mixed_signal[:data_len].reshape(n_items_per_eval, -1) - reconstructed])
            del reconstructed
            del single_source
            del mixed_signal
            gc.collect()
            sdr_b,  sir_b, sar_b, _ =mir_eval.separation.bss_eval_sources_framewise(np.array([reference_music, clean_inference]), np.array([estimates_music, predicted_inference]))
            sdr, sir, sar = sdr_b, sir_b, sar_b
            #sdr_inter, sir_inter, sar_inter = sdr_b[1], sir_b[1], sar_b[1]
            out_sdr.append(np.mean(sdr[~np.isnan(sdr)]))
            out_sir.append(np.mean(sir[~np.isnan(sir)]))
            out_sar.append(np.mean(sir[~np.isnan(sar)]))
        sdr  = np.median(out_sdr)
        sir = np.median(out_sir)
        sar = np.median(out_sar)
        
        results = {
            "sdr": sdr,
            "sir": sir,
            "sar": sar
            }
        return results