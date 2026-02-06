"""
Compound-opcode benchmark for ECA reservoirs.

Task:
  - Packet 1: 4-bit opcode op1 + operands a,b
  - Packet 2: 4-bit opcode op2 + operand c
  - Label:
        y1 = f_op1(a,b)
        y  = f_op2(y1,c)

Opcode definition:
  - Bits are MSB-first [op3, op2, op1, op0].
  - The opcode encodes the full truth table for f(x, y) with ordering:
        (x,y) = 00, 01, 10, 11
    op0 is the output for 00, op1 for 01, op2 for 10, op3 for 11.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.svm import SVC

from ..eca import eca_rule_lkt, eca_step
from ..utils import create_input_locations, flatten_history


# -----------------------------
# Channel layout (15 channels)
# -----------------------------
TAG_OP1_0 = 0
TAG_OP1_1 = 1
TAG_OP1_2 = 2
TAG_OP1_3 = 3
TAG_A = 4
TAG_B = 5

TAG_OP2_0 = 6
TAG_OP2_1 = 7
TAG_OP2_2 = 8
TAG_OP2_3 = 9
TAG_C = 10

VAL_0 = 11
VAL_1 = 12
DIST = 13
CUE = 14

N_CHANNELS = 15

TAG_CHANNELS = [
    TAG_OP1_0,
    TAG_OP1_1,
    TAG_OP1_2,
    TAG_OP1_3,
    TAG_A,
    TAG_B,
    TAG_OP2_0,
    TAG_OP2_1,
    TAG_OP2_2,
    TAG_OP2_3,
    TAG_C,
]
VAL_CHANNELS = [VAL_0, VAL_1]


def opcode_to_int(op_bits_msb_first: np.ndarray) -> int:
    """Convert 4 opcode bits in MSB-first order [op3, op2, op1, op0] to int 0..15."""
    b = op_bits_msb_first.astype(int).tolist()
    return (b[0] << 3) | (b[1] << 2) | (b[2] << 1) | b[3]


def apply_opcode(op_bits_msb_first: np.ndarray, x: int, y: int) -> int:
    """Apply the opcode-selected Boolean operation and return 0/1.

    Truth table order: (x,y) = 00, 01, 10, 11.
    """
    op = opcode_to_int(op_bits_msb_first)
    x = int(x) & 1
    y = int(y) & 1
    idx = (x << 1) | y
    return (op >> idx) & 1


def apply_compound_opcode(
    op1_bits_msb_first: np.ndarray,
    a: int,
    b: int,
    op2_bits_msb_first: np.ndarray,
    c: int,
) -> int:
    """Compute y = f_op2(f_op1(a,b), c)."""
    y1 = apply_opcode(op1_bits_msb_first, a, b)
    return apply_opcode(op2_bits_msb_first, y1, c)


def make_packet(tag_channel: int, value: int) -> np.ndarray:
    """Create one 15-channel packet with given tag channel and bit value."""
    pkt = np.zeros(N_CHANNELS, dtype=np.int8)
    pkt[tag_channel] = 1
    if int(value) == 0:
        pkt[VAL_0] = 1
    else:
        pkt[VAL_1] = 1
    return pkt


def build_tagged_stream(
    op1_bits_msb_first: np.ndarray,
    a: int,
    b: int,
    op2_bits_msb_first: np.ndarray,
    c: int,
    d_period: int,
    repeats: int = 1,
) -> np.ndarray:
    """Build the input stream of shape (L, 15).

    Fixed write order:
      op1 bits (op1_0..op1_3), a, b, op2 bits (op2_0..op2_3), c
    """
    op1_bits_msb_first = np.asarray(op1_bits_msb_first, dtype=np.int8).reshape(-1)
    op2_bits_msb_first = np.asarray(op2_bits_msb_first, dtype=np.int8).reshape(-1)
    if op1_bits_msb_first.size != 4:
        raise ValueError("op1_bits_msb_first must have length 4 (op3, op2, op1, op0).")
    if op2_bits_msb_first.size != 4:
        raise ValueError("op2_bits_msb_first must have length 4 (op3, op2, op1, op0).")

    field_to_tag_and_value = {
        "op1_0": (TAG_OP1_0, int(op1_bits_msb_first[3])),
        "op1_1": (TAG_OP1_1, int(op1_bits_msb_first[2])),
        "op1_2": (TAG_OP1_2, int(op1_bits_msb_first[1])),
        "op1_3": (TAG_OP1_3, int(op1_bits_msb_first[0])),
        "a": (TAG_A, int(a)),
        "b": (TAG_B, int(b)),
        "op2_0": (TAG_OP2_0, int(op2_bits_msb_first[3])),
        "op2_1": (TAG_OP2_1, int(op2_bits_msb_first[2])),
        "op2_2": (TAG_OP2_2, int(op2_bits_msb_first[1])),
        "op2_3": (TAG_OP2_3, int(op2_bits_msb_first[0])),
        "c": (TAG_C, int(c)),
    }

    order = [
        "op1_0",
        "op1_1",
        "op1_2",
        "op1_3",
        "a",
        "b",
        "op2_0",
        "op2_1",
        "op2_2",
        "op2_3",
        "c",
    ]

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

    # Set DIST=1 by default on all non-cue ticks (including write packets)
    stream[:-1, DIST] = 1
    stream[-1, DIST] = 0

    return stream


def run_episode_record_tagged(
    op1_bits_msb_first: np.ndarray,
    a: int,
    b: int,
    op2_bits_msb_first: np.ndarray,
    c: int,
    rule_number: int,
    width: int,
    boundary: str,
    itr: int,
    d_period: int,
    rng: np.random.Generator,
    input_locations: np.ndarray,
    repeats: int = 1,
    reg: Optional[SVC] = None,
    collect_states: bool = True,
    x0_mode: str = "zeros",
    feature_mode: str = "cue_tick",
    output_window: int = 2,
) -> dict:
    """Run one compound-opcode episode and (optionally) score with a readout.

    feature_mode:
      - "cue_tick": use features at the cue tick only (one sample per episode)
      - "output_window": use concatenated features over the last `output_window` ticks
                         (returns X_out shape (output_window, itr*width))
    """
    rule = eca_rule_lkt(rule_number)

    input_streams = build_tagged_stream(
        op1_bits_msb_first=op1_bits_msb_first,
        a=a,
        b=b,
        op2_bits_msb_first=op2_bits_msb_first,
        c=c,
        d_period=d_period,
        repeats=repeats,
    )

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

    # True label is the compound gate output (1 bit) at episode level
    y_true = apply_compound_opcode(op1_bits_msb_first, a, b, op2_bits_msb_first, c)

    # Package episode-level features
    if feature_mode == "cue_tick":
        X_episode = X_tick[-1]  # cue tick is the last tick
    elif feature_mode == "output_window":
        if output_window < 1:
            raise ValueError("output_window must be >= 1")
        X_episode = X_tick[-output_window:]  # shape (W, itr*width)
    else:
        raise ValueError("feature_mode must be 'cue_tick' or 'output_window'")

    return {
        "inputs_tick": inputs_tick,
        "X_tick": X_tick,
        "X_episode": X_episode,
        "y_true": int(y_true),
        "y_pred_tick": y_pred,
        "states": states,
        "L": L,
        "T": T,
        "iter_between": iter_between,
        "input_streams": input_streams,
    }


def build_dataset_compound_opcode(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate all combinations and return (X, y, input_locations)."""
    rng = np.random.default_rng(seed)
    input_locations = create_input_locations(width, recurrence, N_CHANNELS, rng)

    X_all = []
    y_all = []

    for op1_int in range(16):
        op1_bits = np.array(
            [(op1_int >> 3) & 1, (op1_int >> 2) & 1, (op1_int >> 1) & 1, op1_int & 1],
            dtype=np.int8,
        )  # op3,op2,op1,op0
        for a in (0, 1):
            for b in (0, 1):
                for op2_int in range(16):
                    op2_bits = np.array(
                        [
                            (op2_int >> 3) & 1,
                            (op2_int >> 2) & 1,
                            (op2_int >> 1) & 1,
                            op2_int & 1,
                        ],
                        dtype=np.int8,
                    )  # op3,op2,op1,op0
                    for c in (0, 1):
                        ep = run_episode_record_tagged(
                            op1_bits_msb_first=op1_bits,
                            a=a,
                            b=b,
                            op2_bits_msb_first=op2_bits,
                            c=c,
                            rule_number=rule_number,
                            width=width,
                            boundary=boundary,
                            itr=itr,
                            d_period=d_period,
                            rng=rng,
                            input_locations=input_locations,
                            repeats=repeats,
                            reg=None,
                            collect_states=False,
                            x0_mode="zeros",
                            feature_mode=feature_mode,
                            output_window=output_window,
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


def train_compound_opcode_readout(
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
) -> Tuple[SVC, np.ndarray]:
    """Train a linear SVM readout for the compound-opcode task."""
    X, y, input_locations = build_dataset_compound_opcode(
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
    )
    reg = SVC(kernel="linear")
    reg.fit(X, y)
    return reg, input_locations
