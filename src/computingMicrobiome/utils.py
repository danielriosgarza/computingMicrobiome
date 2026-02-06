from __future__ import annotations

from typing import List

import numpy as np


def int_to_bits(i: int, n_bits: int) -> np.ndarray:
    """
    Return np.int8 array of length n_bits, MSB-first.
    Example: 5 -> [1, 0, 1] for n_bits=3.
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    s = np.binary_repr(int(i), width=n_bits)
    return np.array([int(c) for c in s], dtype=np.int8)


def int_to_bits_lsb(i: int, n_bits: int) -> np.ndarray:
    """Return np.int8 array of length n_bits, LSB-first."""
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    bits = [(int(i) >> k) & 1 for k in range(n_bits)]
    return np.array(bits, dtype=np.int8)


def bits_lsb_to_int(bits_lsb: np.ndarray) -> int:
    """Convert LSB-first bit array to integer."""
    bits_lsb = np.asarray(bits_lsb, dtype=np.int8).reshape(-1)
    val = 0
    for k, b in enumerate(bits_lsb.tolist()):
        val |= (int(b) & 1) << k
    return val


def flatten_history(history_list_of_arrays: List[np.ndarray]) -> np.ndarray:
    """Concatenate a list of 1D arrays into a flat feature vector."""
    return np.concatenate(history_list_of_arrays, axis=0)


def create_input_locations(
    width: int,
    recurrence: int,
    input_channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample injection sites, evenly partitioning width into segments.

    Returns an array of length recurrence * input_channels.
    """
    if width < recurrence:
        raise ValueError("width must be >= recurrence")
    single_min = width // recurrence
    rest = width % recurrence
    if input_channels > single_min:
        raise ValueError("input_channels exceeds minimum segment width")

    r_widths = np.full(recurrence, single_min, dtype=int)
    r_widths[:rest] += 1

    locs = []
    offset = 0
    for i in range(recurrence):
        seg_w = r_widths[i]
        seg_positions = rng.choice(seg_w, size=input_channels, replace=False)
        locs.extend((seg_positions + offset).tolist())
        offset += seg_w

    return np.array(locs, dtype=int)
