from computingMicrobiome.benchmarks.k_serial_adder_bm import *  # noqa: F401,F403

__all__ = [
    "N_CHANNELS",
    "build_dataset_serial_adder",
    "build_tagged_stream_serial_adder",
    "run_episode_record_serial_adder",
]
"""
Serial adder benchmark for ECA reservoirs.

Task:
  - Stream bits of two N-bit integers A and B from LSB to MSB.
  - Maintain reservoir state across bits to preserve carry.
  - Label for bit i is the i-th bit of (A + B).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.svm import SVC

from elementaryCA import eca_rule_lkt, eca_step

# -----------------------------
# Channel layout (5 channels)
# -----------------------------
TAG_A = 0
TAG_B = 1
VAL = 2
DIST = 3
CUE = 4

N_CHANNELS = 5


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


def make_packet(tag_channel: int, value: int) -> np.ndarray:
    """Create one 5-channel packet with given tag channel and bit value."""
    pkt = np.zeros(N_CHANNELS, dtype=np.int8)
    pkt[tag_channel] = 1
    if int(value) == 1:
        pkt[VAL] = 1
    return pkt


def build_tagged_stream_serial_adder(
    a_bits_lsb: np.ndarray,
    b_bits_lsb: np.ndarray,
    d_period: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build input stream (L, 5) and cue indices for serial addition."""
    a_bits_lsb = np.asarray(a_bits_lsb, dtype=np.int8).reshape(-1)
    b_bits_lsb = np.asarray(b_bits_lsb, dtype=np.int8).reshape(-1)
    if a_bits_lsb.size != b_bits_lsb.size:
        raise ValueError("a_bits_lsb and b_bits_lsb must have the same length")

    stream_list: List[np.ndarray] = []
    cue_indices: List[int] = []

    for k in range(a_bits_lsb.size):
        # Write A_k, then B_k
        stream_list.append(make_packet(TAG_A, int(a_bits_lsb[k])))
        stream_list.append(make_packet(TAG_B, int(b_bits_lsb[k])))

        # Distractor ticks
        for _ in range(int(d_period)):
            distractor = np.zeros(N_CHANNELS, dtype=np.int8)
            distractor[DIST] = 1
            stream_list.append(distractor)

        # Cue tick for this bit position
        cue = np.zeros(N_CHANNELS, dtype=np.int8)
        cue[CUE] = 1
        cue_indices.append(len(stream_list))
        stream_list.append(cue)

    stream = np.vstack(stream_list)

    # Set DIST=1 by default on all non-cue ticks (including writes)
    stream[:, DIST] = 1
    stream[cue_indices, DIST] = 0

    return stream, np.array(cue_indices, dtype=int)


def create_input_locations(
    width: int,
    recurrence: int,
    input_channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample injection sites by segment, same as other benchmarks."""
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


def flatten_history(history_list_of_arrays: List[np.ndarray]) -> np.ndarray:
    """Concatenate a list of 1D arrays into a flat feature vector."""
    return np.concatenate(history_list_of_arrays, axis=0)


def run_episode_record_serial_adder(
    a: int,
    b: int,
    bits: int,
    rule_number: int,
    width: int,
    boundary: str,
    itr: int,
    d_period: int,
    rng: np.random.Generator,
    input_locations: np.ndarray,
    reg: Optional[SVC] = None,
    collect_states: bool = True,
    x0_mode: str = "zeros",
) -> dict:
    """Run one serial-adder episode and (optionally) score with a readout."""
    rule = eca_rule_lkt(rule_number)

    a_bits_lsb = int_to_bits_lsb(a, bits)
    b_bits_lsb = int_to_bits_lsb(b, bits)
    input_streams, cue_indices = build_tagged_stream_serial_adder(a_bits_lsb, b_bits_lsb, d_period)

    L = input_streams.shape[0]
    iter_between = itr + 1
    T = L * iter_between

    if x0_mode == "zeros":
        x = np.zeros(width, dtype=np.int8)
    elif x0_mode == "random":
        x = rng.integers(0, 2, size=width, dtype=np.int8)
    else:
        raise ValueError("x0_mode must be 'zeros' or 'random'")

    history = [np.zeros(width, dtype=np.int8) for _ in range(itr)]

    inputs_tick = np.zeros((L, N_CHANNELS), dtype=np.int8)
    X_tick = np.zeros((L, itr * width), dtype=np.int8)
    y_pred = np.full(L, -1, dtype=np.int8)

    states = np.zeros((T, width), dtype=np.int8) if collect_states else None

    channel_idx = np.arange(input_locations.size) % N_CHANNELS

    tick = 0
    for i in range(T):
        if i % iter_between == 0:
            in_bits = input_streams[tick]
            inputs_tick[tick] = in_bits

            # XOR inject: each injection site reads one channel
            x[input_locations] ^= in_bits[channel_idx]

        history.append(x.copy())
        history = history[-itr:]

        if i % iter_between == 0:
            feat = flatten_history(history)
            X_tick[tick] = feat
            if reg is not None:
                y_pred[tick] = int(reg.predict([feat])[0])
            tick += 1

        if collect_states:
            states[i] = x

        x = eca_step(x, rule, boundary, rng=rng)

    sum_bits_lsb = int_to_bits_lsb(a + b, bits)

    return {
        "inputs_tick": inputs_tick,
        "X_tick": X_tick,
        "y_pred_tick": y_pred,
        "states": states,
        "L": L,
        "T": T,
        "iter_between": iter_between,
        "input_streams": input_streams,
        "cue_indices": cue_indices,
        "y_true_bits": sum_bits_lsb,
    }


def build_dataset_serial_adder(
    bits: int,
    n_samples: int,
    rule_number: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dataset from random N-bit addition problems."""
    rng = np.random.default_rng(seed)
    input_locations = create_input_locations(width, recurrence, N_CHANNELS, rng)

    X_all = []
    y_all = []

    for _ in range(int(n_samples)):
        a = int(rng.integers(0, 2**bits))
        b = int(rng.integers(0, 2**bits))

        ep = run_episode_record_serial_adder(
            a=a,
            b=b,
            bits=bits,
            rule_number=rule_number,
            width=width,
            boundary=boundary,
            itr=itr,
            d_period=d_period,
            rng=rng,
            input_locations=input_locations,
            reg=None,
            collect_states=False,
            x0_mode="zeros",
        )

        cue_idx = ep["cue_indices"]
        X_out = ep["X_tick"][cue_idx]  # shape (bits, itr*width)
        y_bits = ep["y_true_bits"]     # shape (bits,)

        X_all.append(X_out)
        y_all.append(y_bits)

    X = np.vstack(X_all)
    y = np.concatenate(y_all).astype(np.int8)
    return X, y, input_locations
