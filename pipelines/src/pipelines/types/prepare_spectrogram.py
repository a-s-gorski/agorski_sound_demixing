from pydantic import BaseModel


class SpectrogramConfig(BaseModel):
    """
    Represents a config for calculating spectrogram out of a signal.

    Attributes:
        sample_freq (int): The sample frequency for loading the signal. This determines how many samples are captured per second.
        n_fft (int): The number of samples used in the Fast Fourier Transform (FFT) for calculating the spectrogram. It determines the number of frequency bins in the output.
        time_mask (bool): If True, applies time masking to the spectrogram.
        time_mask_param (int): The size of the time mask window for time masking.
        freq_mask (bool): If True, applies frequency masking to the spectrogram.
        freq_mask_param (int): The size of the frequency mask window for frequency masking.
        to_db (bool): If True, converts the magnitude spectrogram to decibels (dB) for better human perception.
        output_width (int): The desired width of the output spectrogram - for unet has to be a power of 2.
        output_height (int): The desired height of the output spectrogram - for unet has to be a power of 2.
    """
    sample_freq: int
    n_fft: int
    time_mask: bool
    time_mask_param: int
    freq_mask: bool
    freq_mask_param: int
    to_db: bool
    output_width: int
    output_height: int
