"""K-bit memory benchmark on elementary cellular automata.

This module builds input/output streams, injects them into an ECA,
and trains a linear readout on the output window to assess memory.
"""

from typing import Optional, Tuple

import numpy as np

from .episode_runner import run_reservoir_episode
from ..readouts.base import Readout
from ..readouts.factory import make_readout
from ..reservoirs.factory import make_reservoir
from ..utils import create_input_locations, int_to_bits


def true_bits_from_episode_outputs(outputs_tick: np.ndarray, bits: int) -> np.ndarray:
    """Decode target bits from the output streams.

    Args:
        outputs_tick: Output streams of shape (L, 3).
        bits: Number of bits encoded at the end of the episode.

    Returns:
        np.ndarray: Bit vector of shape (bits,) with values in {0, 1}.
    """
    out = outputs_tick[-bits:]
    true = np.full(bits, -1, dtype=np.int8)
    true[out[:, 0] == 1] = 1
    true[out[:, 1] == 1] = 0
    return true


def create_input_streams(bits_arr: np.ndarray, d_period: int) -> np.ndarray:
    """Create input streams for the memory task.

    Args:
        bits_arr: Bit vector of shape (bits,).
        d_period: Delay period between input and recall window.

    Returns:
        np.ndarray: Input streams of shape (L, 4) where
        L = d_period + 2 * bits.
    """

    B = len(bits_arr)
    L = d_period + 2 * B
    streams = np.zeros((L, 4), dtype=np.int8)

    # ch0: bits
    streams[:B, 0] = bits_arr

    # ch1: flipped bits
    streams[:B, 1] = np.bitwise_xor(bits_arr, 1)

    # ch2: distractor
    cue_idx = L - B - 1
    streams[:, 2] = 1
    streams[:B, 2] = 0
    streams[cue_idx, 2] = 0

    # ch3: cue
    streams[cue_idx, 3] = 1

    return streams


def create_output_streams(bits_arr: np.ndarray, d_period: int) -> np.ndarray:
    """Create output streams for the memory task.

    Args:
        bits_arr: Bit vector of shape (bits,).
        d_period: Delay period between input and recall window.

    Returns:
        np.ndarray: Output streams of shape (L, 3) where
        L = d_period + 2 * bits.
    """

    B = len(bits_arr)
    L = d_period + 2 * B
    outs = np.zeros((L, 3), dtype=np.int8)

    outs[-B:, 0] = bits_arr
    outs[-B:, 1] = np.bitwise_xor(bits_arr, 1)

    outs[:, 2] = 1
    outs[-B:, 2] = 0

    return outs


def label_from_outputs(triple: np.ndarray) -> int:
    """Map [y0, y1, y2] to class 0/1/2.

    Args:
        triple: Output triple of shape (3,).

    Returns:
        int: Class label (0, 1, or 2).
    """
    if triple[0] == 1:
        return 0
    if triple[1] == 1:
        return 1
    return 2


def run_episode_record(
    bits_arr: np.ndarray,
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
    """Run one episode and optionally record states and predictions.

    Args:
        bits_arr: Bit vector of shape (bits,).
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        itr: Number of iterations between ticks.
        d_period: Delay period between input and recall.
        rng: NumPy random generator.
        input_locations: Injection locations array.
        reg: Optional trained readout model for predictions.
        collect_states: Whether to store the full state history.
        x0_mode: Initial state mode ("zeros" or "random").

    Returns:
        dict: Episode data including inputs, outputs, features, and predictions.
    """
    B = len(bits_arr)
    L = d_period + 2 * B

    input_streams = create_input_streams(bits_arr, d_period)
    output_streams = create_output_streams(bits_arr, d_period)
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

    y_true = np.array([label_from_outputs(row) for row in output_streams], dtype=np.int8)

    return {
        "inputs_tick": ep["inputs_tick"],
        "outputs_tick": output_streams,
        "y_true": y_true,
        "X_tick": ep["X_tick"],
        "y_pred": ep["y_pred_tick"],
        "states": ep["states"],
        "L": ep["L"],
        "T": ep["T"],
        "iter_between": ep["iter_between"],
    }


def build_dataset_output_window_only(
    bits: int,
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
    """Build a dataset using only the output window.

    Args:
        bits: Number of bits to store/recall.
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Delay period between input and recall.
        seed: RNG seed.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Feature matrix, labels,
        and input locations.
    """
    rng = np.random.default_rng(seed)
    input_locations = create_input_locations(width, recurrence, 4, rng)

    X_all = []
    y_all = []

    for val in range(2**bits):
        bits_arr = int_to_bits(val, bits)

        ep = run_episode_record(
            bits_arr=bits_arr,
            rule_number=rule_number,
            width=width,
            boundary=boundary,
            itr=itr,
            d_period=d_period,
            rng=rng,
            input_locations=input_locations,
            reg=None,
            collect_states=False,
            x0_mode="zeros",  # IMPORTANT
            reservoir_kind=reservoir_kind,
            reservoir_config=reservoir_config,
        )

        # output window samples (last bits ticks)
        X_out = ep["X_tick"][-bits:]  # shape (bits, itr*width)
        true_bits = true_bits_from_episode_outputs(ep["outputs_tick"], bits)

        # true_bits is guaranteed 0/1 in output window
        X_all.append(X_out)
        y_all.append(true_bits)

    X = np.vstack(X_all)
    y = np.concatenate(y_all).astype(np.int8)
    return X, y, input_locations


def train_memory_readout(
    bits: int,
    rule_number: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    seed_train: int = 0,
    readout_kind: str = "svm",
    readout_config: Optional[dict] = None,
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
) -> Tuple[Readout, np.ndarray]:
    """Train a linear readout on the output-window dataset.

    Args:
        bits: Number of bits to store/recall.
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Delay period between input and recall.
        seed_train: RNG seed for training data.
        readout_kind: "svm" or "evo".
        readout_config: Optional configuration for the readout.

    Returns:
        Tuple[Readout, np.ndarray]: Trained classifier and input locations.
    """
    X, y, input_locations = build_dataset_output_window_only(
        bits,
        rule_number,
        width,
        boundary,
        recurrence,
        itr,
        d_period,
        seed=seed_train,
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
    )
    rng = np.random.default_rng(seed_train)
    reg = make_readout(readout_kind, readout_config, rng=rng)
    reg.fit(X, y)
    print("Training set shape:", X.shape, "labels:", np.unique(y))
    print("Training accuracy (output-window only):", reg.score(X, y))
    return reg, input_locations


def evaluate_memory_trials(
    reg: Readout,
    bits: int,
    rule_number: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    input_locations: np.ndarray,
    n_trials: int = 100,
    seed_trials: int = 0,
    resample_input_locations: bool = False,
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
) -> np.ndarray:
    """Evaluate recall correctness across randomized trials.

    Args:
        reg: Trained readout model.
        bits: Number of bits.
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Delay period between input and recall.
        input_locations: Injection locations array.
        n_trials: Number of random trials.
        seed_trials: RNG seed for trials.
        resample_input_locations: If True, resample locations per trial.

    Returns:
        np.ndarray: Correctness matrix of shape (n_trials, bits) with values
        in {-1, 1}.
    """
    rng = np.random.default_rng(seed_trials)

    correctness = np.empty((n_trials, bits), dtype=np.int8)

    for k in range(n_trials):
        val = int(rng.integers(0, 2**bits))
        bits_arr = int_to_bits(val, bits)

        locs = (
            create_input_locations(width, recurrence, 4, rng)
            if resample_input_locations
            else input_locations
        )

        ep = run_episode_record(
            bits_arr=bits_arr,
            rule_number=rule_number,
            width=width,
            boundary=boundary,
            itr=itr,
            d_period=d_period,
            rng=rng,
            input_locations=locs,
            reg=None,
            collect_states=False,
            x0_mode="zeros",
            reservoir_kind=reservoir_kind,
            reservoir_config=reservoir_config,
        )

        X_out = ep["X_tick"][-bits:]  # features in output window
        pred_bits = reg.predict(X_out).astype(np.int8)
        true_bits = true_bits_from_episode_outputs(ep["outputs_tick"], bits).astype(
            np.int8
        )

        correctness[k] = np.where(pred_bits == true_bits, 1, -1)

    return correctness
