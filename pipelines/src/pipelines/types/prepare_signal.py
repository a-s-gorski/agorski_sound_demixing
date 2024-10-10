from pydantic import BaseModel


class PrepareSignalConfig(BaseModel):
    """
    Represents a config for processing signal.

    Attributes:
        sample_freq (int): sample frequency for loading signal
        normalize (bool): whether to apply normalization
        highpass (bool): whether to apply highpass filter
        highpass_freq (int): highpass frequency if filter is applied
        lowpass (bool): whether to apply lowpass filter
        lowpass_freq (int): lowpass frequency if low pass filter is applied
        resample (bool): whether to resample signal after applying filters
        resample_freq: (int): resampling frequency
        speed (bool): whether to apply speed transformation
        speed_ratio (float): float below 1 extends signal, above 1 shortens it
        reverb (bool): whether to apply reverb
        channels (bool): whether to modify number of channels
        channels_num (int): how many output channels are desired - recommended 1 for mono, 2 for stereo
        subseq_len (int): desired length of subsequence
    """
    sample_freq: int
    normalize: bool
    highpass: bool
    highpass_freq: int
    lowpass: bool
    lowpass_freq: int
    resample: bool
    resample_freq: int
    speed: bool
    speed_ratio: float
    reverb: bool
    channels: bool
    channels_num: int
    subseq_len: int
