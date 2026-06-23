import re
from typing import Tuple


def version_to_tuple(version: str) -> Tuple[int, ...]:
    return tuple(int(i) for i in re.findall(r"\d+", version))


def parse_channel_pair(channel_name: str, raw_channels) -> Tuple[int, int]:
    if not isinstance(raw_channels, (tuple, list)) or len(raw_channels) != 2:
        raise ValueError(
            f"Invalid channel mapping for '{channel_name}': expected a pair [pos, neg], got {raw_channels}."
        )
    return int(raw_channels[0]), int(raw_channels[1])


def infer_num_interaction_channels_from_mapping(channel_mapping: dict) -> int:
    max_positive_index = -1
    max_negative_magnitude = 0

    for k, v in channel_mapping.items():
        if k == "prev_seg":
            indices = [int(v)]
        else:
            pos_ch, neg_ch = parse_channel_pair(k, v)
            indices = [pos_ch, neg_ch]

        for idx in indices:
            if idx >= 0:
                max_positive_index = max(max_positive_index, idx)
            else:
                max_negative_magnitude = max(max_negative_magnitude, abs(idx))

    # Positive indexing is 0-based, while negative indexing is 1-based-from-end.
    return max(max_positive_index + 1, max_negative_magnitude, 1)
