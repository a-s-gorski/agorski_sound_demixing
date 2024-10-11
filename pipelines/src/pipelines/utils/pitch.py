from typing import Dict

from pipelines.utils import get_pitch_shift_factor


def get_pitch_shifted_segment_samples(segment_samples: int, augmentations: Dict) -> int:
    r"""Get new segment samples depending on maximum pitch shift.

    Args:
        segment_samples: int
        augmentations: Dict

    Returns:
        ex_segment_samples: int
    """

    if 'pitch_shift' not in augmentations.keys():
        return segment_samples

    else:
        pitch_shift_dict = augmentations['pitch_shift']
        source_types = pitch_shift_dict.keys()

    max_pitch_shift = max(
        [pitch_shift_dict[source_type] for source_type in source_types]
    )

    ex_segment_samples = int(segment_samples * get_pitch_shift_factor(max_pitch_shift))

    return ex_segment_samples
