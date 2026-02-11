"""Programmable-logic benchmark for ECA reservoirs.

Task:
  - 3-bit opcode selects a Boolean operation to apply to operands (a, b).
  - Inputs are delivered as *tagged packets* using one-hot tag channels + value channels.
  - The reservoir is driven deterministically (no randomness required beyond choosing injection sites).
  - A linear readout (e.g., SVM) is trained to predict the 1-bit gate output at the cue.

Packet encoding (one input event = one tick in the input-stream):
  - Exactly one TAG_* channel is 1, indicating which field is being written.
  - Exactly one VAL_* channel is 1, indicating the bit value (0 or 1).
  - DIST channel is 1 on most ticks, and turned off at the CUE tick.
  - CUE channel is 1 only at the cue tick.

Default operations (opcode in MSB-first order op2 op1 op0):
  000 AND
  001 OR
  010 XOR
  011 NAND
  100 NOR
  101 XNOR
  110 A
  111 B
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from .episode_runner import run_reservoir_episode
from ..reservoirs.factory import make_reservoir
from ..utils import create_input_locations
from ..readouts.base import Readout
from ..readouts.factory import make_readout


# -----------------------------
# Channel layout (9 channels)
# -----------------------------
TAG_OP0 = 0
TAG_OP1 = 1
TAG_OP2 = 2
TAG_A = 3
TAG_B = 4
VAL_0 = 5
VAL_1 = 6
DIST = 7
CUE = 8

N_CHANNELS = 9

TAG_CHANNELS = [TAG_OP0, TAG_OP1, TAG_OP2, TAG_A, TAG_B]
VAL_CHANNELS = [VAL_0, VAL_1]


def opcode_to_int(op_bits_msb_first: np.ndarray) -> int:
    """Convert MSB-first opcode bits to an integer in [0, 7].

    Args:
        op_bits_msb_first: Array of shape (3,) in order [op2, op1, op0].

    Returns:
        int: Opcode integer in [0, 7].
    """
    b = op_bits_msb_first.astype(int).tolist()
    return (b[0] << 2) | (b[1] << 1) | b[2]


def apply_opcode(op_bits_msb_first: np.ndarray, a: int, b: int) -> int:
    """Apply the opcode-selected Boolean operation.

    Args:
        op_bits_msb_first: Array of shape (3,) in order [op2, op1, op0].
        a: Operand bit (0 or 1).
        b: Operand bit (0 or 1).

    Returns:
        int: Result bit (0 or 1).
    """
    op = opcode_to_int(op_bits_msb_first)
    a = int(a) & 1
    b = int(b) & 1

    if op == 0:  # AND
        return a & b
    if op == 1:  # OR
        return a | b
    if op == 2:  # XOR
        return a ^ b
    if op == 3:  # NAND
        return 1 - (a & b)
    if op == 4:  # NOR
        return 1 - (a | b)
    if op == 5:  # XNOR
        return 1 - (a ^ b)
    if op == 6:  # A
        return a
    if op == 7:  # B
        return b
    raise ValueError("Opcode out of range (should be 0..7).")


def make_packet(tag_channel: int, value: int) -> np.ndarray:
    """Create a single tagged packet.

    Args:
        tag_channel: Tag channel index.
        value: Bit value (0 or 1).

    Returns:
        np.ndarray: Packet of shape (9,) with dtype int8.
    """
    pkt = np.zeros(N_CHANNELS, dtype=np.int8)
    pkt[tag_channel] = 1
    if int(value) == 0:
        pkt[VAL_0] = 1
    else:
        pkt[VAL_1] = 1
    return pkt


def build_tagged_stream(
    op_bits_msb_first: np.ndarray,
    a: int,
    b: int,
    d_period: int,
    repeats: int = 1,
    order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Build the tagged input stream for a single episode.

    Args:
        op_bits_msb_first: Array of shape (3,) in order [op2, op1, op0].
        a: Operand bit (0 or 1).
        b: Operand bit (0 or 1).
        d_period: Distractor length in input ticks.
        repeats: Number of times to repeat the write packets.
        order: Optional sequence of field names among {"op0", "op1", "op2", "a", "b"}.
            If None, uses ["op0", "op1", "op2", "a", "b"].

    Returns:
        np.ndarray: Stream of shape (L, 9).
    """
    op_bits_msb_first = np.asarray(op_bits_msb_first, dtype=np.int8).reshape(-1)
    if op_bits_msb_first.size != 3:
        raise ValueError("op_bits_msb_first must have length 3 (op2, op1, op0).")

    if order is None:
        order = ["op0", "op1", "op2", "a", "b"]

    field_to_tag_and_value = {
        "op0": (TAG_OP0, int(op_bits_msb_first[2])),
        "op1": (TAG_OP1, int(op_bits_msb_first[1])),
        "op2": (TAG_OP2, int(op_bits_msb_first[0])),
        "a": (TAG_A, int(a)),
        "b": (TAG_B, int(b)),
    }

    write_packets: List[np.ndarray] = []
    for _ in range(int(repeats)):
        for key in order:
            tag, val = field_to_tag_and_value[key]
            write_packets.append(make_packet(tag, val))

    # Distractor ticks: only DIST=1 (no tags, no values)
    distractor = np.zeros((int(d_period), N_CHANNELS), dtype=np.int8)
    if d_period > 0:
        distractor[:, DIST] = 1

    # Cue tick: CUE=1, DIST=0
    cue = np.zeros((1, N_CHANNELS), dtype=np.int8)
    cue[0, CUE] = 1

    stream = np.vstack([np.vstack(write_packets), distractor, cue])

    # Set DIST=1 by default on all non-cue ticks unless during write packets you prefer otherwise.
    # Here we keep DIST=1 during write packets too (often improves separability),
    # but it is harmless because tags + values still exist.
    stream[:-1, DIST] = 1
    stream[-1, DIST] = 0

    return stream


def run_episode_record_tagged(
    op_bits_msb_first: np.ndarray,
    a: int,
    b: int,
    rule_number: int,
    width: int,
    boundary: str,
    itr: int,
    d_period: int,
    rng: np.random.Generator,
    input_locations: np.ndarray,
    repeats: int = 1,
    order: Optional[Sequence[str]] = None,
    reg: Optional[Readout] = None,
    collect_states: bool = True,
    x0_mode: str = "zeros",
    feature_mode: str = "cue_tick",
    output_window: int = 2,
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
) -> dict:
    """Run one tagged-program episode and optionally score with a readout.

    Args:
        op_bits_msb_first: Array of shape (3,) in order [op2, op1, op0].
        a: Operand bit (0 or 1).
        b: Operand bit (0 or 1).
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        itr: Number of iterations between ticks.
        d_period: Distractor length in input ticks.
        rng: NumPy random generator.
        input_locations: Injection locations array.
        repeats: Number of times to repeat the write packets.
        order: Optional packet field ordering.
        reg: Optional trained readout model for predictions.
        collect_states: Whether to store the full state history.
        x0_mode: Initial state mode ("zeros" or "random").
        feature_mode: "cue_tick" or "output_window".
        output_window: Window length when feature_mode is "output_window".

    Returns:
        dict: Episode data including inputs, features, predictions, and states.
    """
    input_streams = build_tagged_stream(
        op_bits_msb_first=op_bits_msb_first,
        a=a,
        b=b,
        d_period=d_period,
        repeats=repeats,
        order=order,
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

    # True label is the gate output (1 bit) at episode level
    y_true = apply_opcode(op_bits_msb_first, a, b)

    # Package episode-level features
    if feature_mode == "cue_tick":
        X_episode = ep["X_tick"][-1]  # cue tick is the last tick
    elif feature_mode == "output_window":
        if output_window < 1:
            raise ValueError("output_window must be >= 1")
        X_episode = ep["X_tick"][-output_window:]  # shape (W, itr*width)
    else:
        raise ValueError("feature_mode must be 'cue_tick' or 'output_window'")

    return {
        "inputs_tick": ep["inputs_tick"],
        "X_tick": ep["X_tick"],
        "X_episode": X_episode,
        "y_true": int(y_true),
        "y_pred_tick": ep["y_pred_tick"],
        "states": ep["states"],
        "L": ep["L"],
        "T": ep["T"],
        "iter_between": ep["iter_between"],
        "input_streams": input_streams,
    }


def build_dataset_programmed_logic(
    rule_number: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    repeats: int = 1,
    feature_mode: str = "cue_tick",
    output_window: int = 2,
    seed: int = 0,
    order: Optional[Sequence[str]] = None,
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dataset for the programmed-logic task.

    Args:
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Distractor length in input ticks.
        repeats: Number of times to repeat write packets.
        feature_mode: "cue_tick" or "output_window".
        output_window: Window length when feature_mode is "output_window".
        seed: RNG seed for dataset generation.
        order: Optional packet field ordering.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Feature matrix, labels,
        and input locations.
    """
    rng = np.random.default_rng(seed)
    input_locations = create_input_locations(width, recurrence, N_CHANNELS, rng)

    X_all = []
    y_all = []

    for op_int in range(8):
        op_bits = np.array(
            [(op_int >> 2) & 1, (op_int >> 1) & 1, op_int & 1], dtype=np.int8
        )  # op2,op1,op0
        for a in (0, 1):
            for b in (0, 1):
                ep = run_episode_record_tagged(
                    op_bits_msb_first=op_bits,
                    a=a,
                    b=b,
                    rule_number=rule_number,
                    width=width,
                    boundary=boundary,
                    itr=itr,
                    d_period=d_period,
                    rng=rng,
                    input_locations=input_locations,
                    repeats=repeats,
                    order=order,
                    reg=None,
                    collect_states=False,
                    x0_mode="zeros",
                    feature_mode=feature_mode,
                    output_window=output_window,
                    reservoir_kind=reservoir_kind,
                    reservoir_config=reservoir_config,
                )
                if feature_mode == "cue_tick":
                    X_all.append(ep["X_episode"][None, :])  # (1, feat)
                else:
                    # output_window provides W samples; concatenate them to one feature vector
                    X_all.append(ep["X_episode"].reshape(1, -1))
                y_all.append(ep["y_true"])

    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int8)
    return X, y, input_locations


def train_programmed_logic_readout(
    rule_number: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    repeats: int = 1,
    feature_mode: str = "cue_tick",
    output_window: int = 2,
    seed_train: int = 0,
    order: Optional[Sequence[str]] = None,
    readout_kind: str = "svm",
    readout_config: Optional[dict] = None,
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
) -> Tuple[Readout, np.ndarray]:
    """Train a linear readout for the programmed-logic task.

    Args:
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition.
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Distractor length in input ticks.
        repeats: Number of times to repeat write packets.
        feature_mode: "cue_tick" or "output_window".
        output_window: Window length when feature_mode is "output_window".
        seed_train: RNG seed for training data.
        order: Optional packet field ordering.
        readout_kind: "svm" or "evo".
        readout_config: Optional configuration for the readout.

    Returns:
        Tuple[Readout, np.ndarray]: Trained classifier and input locations.
    """
    X, y, input_locations = build_dataset_programmed_logic(
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        recurrence=recurrence,
        itr=itr,
        d_period=d_period,
        repeats=repeats,
        feature_mode=feature_mode,
        output_window=output_window,
        seed=seed_train,
        order=order,
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
    )
    rng = np.random.default_rng(seed_train)
    reg = make_readout(readout_kind, readout_config, rng=rng)
    reg.fit(X, y)
    return reg, input_locations
