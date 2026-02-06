"""K-bit memory benchmark on elementary cellular automata.

This module builds input/output streams, injects them into an ECA,
and trains a linear readout on the output window to assess memory.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.svm import SVC

from ..eca import eca_rule_lkt, eca_step
from ..utils import create_input_locations, flatten_history, int_to_bits


def true_bits_from_episode_outputs(outputs_tick: np.ndarray, bits: int) -> np.ndarray:
    """
    True bits are encoded in outputs_tick[-bits:]:
        y0=1 -> bit=1
        y1=1 -> bit=0
    """
    out = outputs_tick[-bits:]
    true = np.full(bits, -1, dtype=np.int8)
    true[out[:, 0] == 1] = 1
    true[out[:, 1] == 1] = 0
    return true


def create_input_streams(bits_arr: np.ndarray, d_period: int) -> np.ndarray:
    """
    Returns array shape (L, 4), with L = d_period + 2*B where B=len(bits_arr).
    Channel 0: bits at start
    Channel 1: flipped bits at start
    Channel 2: distractor (mostly 1s), but zeros during write and cue
    Channel 3: cue (single 1 one step before output window)
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
    """
    Returns array shape (L, 3) with L = d_period + 2*B.
    Stream 0: bits in the last B positions
    Stream 1: flipped bits in the last B positions
    Stream 2: "none/distractor": ones except forced to 0 in last B positions
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
    """Map [y0,y1,y2] -> class 0/1/2 exactly like the original."""
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
    reg: Optional[SVC] = None,
    collect_states: bool = True,
    x0_mode: str = "zeros",
) -> dict:
    """
    Run one episode and optionally record states and readout predictions.

    Returns a dict of arrays including inputs/outputs per tick, features,
    predictions, and (optionally) full ECA states over time.
    """
    rule = eca_rule_lkt(rule_number)

    B = len(bits_arr)
    L = d_period + 2 * B
    iter_between = itr + 1
    T = L * iter_between

    input_streams = create_input_streams(bits_arr, d_period)
    output_streams = create_output_streams(bits_arr, d_period)

    if x0_mode == "zeros":
        x = np.zeros(width, dtype=np.int8)
    elif x0_mode == "random":
        x = rng.integers(0, 2, size=width, dtype=np.int8)
    else:
        raise ValueError("x0_mode must be 'zeros' or 'random'")

    history = [np.zeros(width, dtype=np.int8) for _ in range(itr)]

    inputs_tick = np.zeros((L, 4), dtype=np.int8)
    outputs_tick = np.zeros((L, 3), dtype=np.int8)
    y_true = np.zeros(L, dtype=np.int8)
    X_tick = np.zeros((L, itr * width), dtype=np.int8)
    y_pred = np.full(L, -1, dtype=np.int8)

    states = np.zeros((T, width), dtype=np.int8) if collect_states else None

    tick = 0
    channel_idx = np.arange(input_locations.size) % 4
    for i in range(T):
        if i % iter_between == 0:
            in_bits = input_streams[tick]
            out_bits = output_streams[tick]

            inputs_tick[tick] = in_bits
            outputs_tick[tick] = out_bits

            # XOR injection (vectorized by channel index)
            x[input_locations] ^= in_bits[channel_idx]

        history.append(x.copy())
        history = history[-itr:]

        if i % iter_between == 0:
            feat = flatten_history(history)
            X_tick[tick] = feat
            cls = label_from_outputs(out_bits)
            y_true[tick] = cls

            if reg is not None:
                y_pred[tick] = int(reg.predict([feat])[0])

            tick += 1

        if collect_states:
            states[i] = x

        x = eca_step(x, rule, boundary, rng=rng)

    return {
        "inputs_tick": inputs_tick,
        "outputs_tick": outputs_tick,
        "y_true": y_true,
        "X_tick": X_tick,
        "y_pred": y_pred,
        "states": states,
        "L": L,
        "T": T,
        "iter_between": iter_between,
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a dataset using only the output window.

    Returns:
        X: (2^bits * bits, itr*width)
        y: (2^bits * bits,) with values 0/1
        input_locations: fixed injection sites
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
) -> Tuple[SVC, np.ndarray]:
    """Train a linear SVM readout on the output window dataset."""
    X, y, input_locations = build_dataset_output_window_only(
        bits,
        rule_number,
        width,
        boundary,
        recurrence,
        itr,
        d_period,
        seed=seed_train,
    )
    reg = SVC(kernel="linear")
    reg.fit(X, y)
    print("Training set shape:", X.shape, "labels:", np.unique(y))
    print("Training accuracy (output-window only):", reg.score(X, y))
    return reg, input_locations


def evaluate_memory_trials(
    reg: SVC,
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
) -> np.ndarray:
    """Return correctness matrix (n_trials, bits) with values +1/-1."""
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
        )

        X_out = ep["X_tick"][-bits:]  # features in output window
        pred_bits = reg.predict(X_out).astype(np.int8)
        true_bits = true_bits_from_episode_outputs(ep["outputs_tick"], bits).astype(
            np.int8
        )

        correctness[k] = np.where(pred_bits == true_bits, 1, -1)

    return correctness
