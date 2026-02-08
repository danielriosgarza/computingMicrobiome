"""Shared utility helpers for bit encodings and input placement."""

from __future__ import annotations

from typing import List

import numpy as np


def int_to_bits(i: int, n_bits: int) -> np.ndarray:
    """Convert an integer to an MSB-first bit vector.

    Args:
        i: Integer value to convert.
        n_bits: Number of bits in the output vector.

    Returns:
        np.ndarray: Bit vector of shape (n_bits,) with dtype int8.

    Example:
        >>> int_to_bits(5, 3).tolist()
        [1, 0, 1]
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    s = np.binary_repr(int(i), width=n_bits)
    return np.array([int(c) for c in s], dtype=np.int8)


def int_to_bits_lsb(i: int, n_bits: int) -> np.ndarray:
    """Convert an integer to an LSB-first bit vector.

    Args:
        i: Integer value to convert.
        n_bits: Number of bits in the output vector.

    Returns:
        np.ndarray: Bit vector of shape (n_bits,) with dtype int8.
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    bits = [(int(i) >> k) & 1 for k in range(n_bits)]
    return np.array(bits, dtype=np.int8)


def bits_lsb_to_int(bits_lsb: np.ndarray) -> int:
    """Convert an LSB-first bit vector to an integer.

    Args:
        bits_lsb: Bit vector of shape (n_bits,) in LSB-first order.

    Returns:
        int: Integer value represented by the bit vector.
    """
    bits_lsb = np.asarray(bits_lsb, dtype=np.int8).reshape(-1)
    val = 0
    for k, b in enumerate(bits_lsb.tolist()):
        val |= (int(b) & 1) << k
    return val


def flatten_history(history_list_of_arrays: List[np.ndarray]) -> np.ndarray:
    """Concatenate a list of 1D arrays into a flat feature vector.

    Args:
        history_list_of_arrays: List of 1D arrays with matching shapes.

    Returns:
        np.ndarray: Flattened 1D vector formed by concatenation.
    """
    return np.concatenate(history_list_of_arrays, axis=0)


def create_input_locations(
    width: int,
    recurrence: int,
    input_channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample injection sites across the automaton width.

    The width is partitioned into `recurrence` segments and `input_channels`
    positions are sampled without replacement in each segment.

    Args:
        width: Total number of cells.
        recurrence: Number of segments to partition the width into.
        input_channels: Number of channels to inject per segment.
        rng: NumPy random generator for sampling.

    Returns:
        np.ndarray: Integer array of length `recurrence * input_channels`.
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
