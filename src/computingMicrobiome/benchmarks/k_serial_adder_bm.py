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

from .episode_runner import run_reservoir_episode
from ..readouts.base import Readout
from ..reservoirs.factory import make_reservoir
from ..utils import create_input_locations, int_to_bits_lsb

# -----------------------------
# Channel layout (5 channels)
# -----------------------------
TAG_A = 0
TAG_B = 1
VAL = 2
DIST = 3
CUE = 4

N_CHANNELS = 5


def make_packet(tag_channel: int, value: int) -> np.ndarray:
    """Create a single tagged packet.

    Args:
        tag_channel: Tag channel index.
        value: Bit value (0 or 1).

    Returns:
        np.ndarray: Packet of shape (5,) with dtype int8.
    """
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
    """Build input stream and cue indices for serial addition.

    Args:
        a_bits_lsb: LSB-first bits for A.
        b_bits_lsb: LSB-first bits for B.
        d_period: Distractor length in input ticks.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Stream of shape (L, 5) and cue indices.
    """
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
    reg: Optional[Readout] = None,
    collect_states: bool = True,
    x0_mode: str = "zeros",
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
) -> dict:
    """Run one serial-adder episode and optionally score with a readout.

    Args:
        a: Integer A (0 <= a < 2**bits).
        b: Integer B (0 <= b < 2**bits).
        bits: Bit-width for A and B.
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        itr: Number of iterations between ticks.
        d_period: Distractor length in input ticks.
        rng: NumPy random generator.
        input_locations: Injection locations array.
        reg: Optional trained readout model for predictions.
        collect_states: Whether to store the full state history.
        x0_mode: Initial state mode ("zeros" or "random").

    Returns:
        dict: Episode data including inputs, features, predictions, and states.
    """
    a_bits_lsb = int_to_bits_lsb(a, bits)
    b_bits_lsb = int_to_bits_lsb(b, bits)
    input_streams, cue_indices = build_tagged_stream_serial_adder(
        a_bits_lsb, b_bits_lsb, d_period
    )

    reservoir = make_reservoir(
        reservoir_kind=reservoir_kind,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        reservoir_config=reservoir_config,
    )
    ep = run_reservoir_episode(
        input_streams=input_streams,
        reservoir=reservoir,
        itr=itr,
        input_locations=input_locations,
        rng=rng,
        reg=reg,
        collect_states=collect_states,
        x0_mode=x0_mode,
    )

    sum_bits_lsb = int_to_bits_lsb(a + b, bits)

    return {
        "inputs_tick": ep["inputs_tick"],
        "X_tick": ep["X_tick"],
        "y_pred_tick": ep["y_pred_tick"],
        "states": ep["states"],
        "L": ep["L"],
        "T": ep["T"],
        "iter_between": ep["iter_between"],
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
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dataset from random N-bit addition problems.

    Args:
        bits: Bit-width of the adder.
        n_samples: Number of samples to generate.
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Distractor length in input ticks.
        seed: RNG seed for dataset generation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Feature matrix, labels,
        and input locations.
    """
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
            reservoir_kind=reservoir_kind,
            reservoir_config=reservoir_config,
        )

        cue_idx = ep["cue_indices"]
        X_out = ep["X_tick"][cue_idx]  # shape (bits, itr*width)
        y_bits = ep["y_true_bits"]  # shape (bits,)

        X_all.append(X_out)
        y_all.append(y_bits)

    X = np.vstack(X_all)
    y = np.concatenate(y_all).astype(np.int8)
    return X, y, input_locations
