"""Feature-source helpers for readout comparisons."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .benchmarks.k_bit_memory_bm import build_dataset_output_window_only
from .benchmarks.k_compound_opcode_bm import (
    apply_compound_opcode,
    build_dataset_compound_opcode,
)
from .benchmarks.k_opcode_logic_bm import apply_opcode as apply_opcode_3
from .benchmarks.k_opcode_logic_bm import build_dataset_programmed_logic
from .benchmarks.k_opcode_logic16_bm import apply_opcode as apply_opcode_4
from .benchmarks.k_opcode_logic16_bm import build_dataset_programmed_logic as build_dataset_programmed_logic16
from .benchmarks.k_serial_adder_bm import build_dataset_serial_adder
from .tasks.toy_addition import (
    enumerate_addition_dataset,
    build_reservoir_dataset as build_addition_reservoir_dataset,
)
from .utils import int_to_bits_lsb


def _pack_dataset(
    X: np.ndarray, y: np.ndarray, input_locations: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray | None]]:
    return X, y, {"input_locations": input_locations}


def build_reservoir_opcode_logic_dataset(
    *, reservoir_kind: str = "eca", reservoir_config: dict | None = None, **kwargs
):
    X, y, input_locations = build_dataset_programmed_logic(
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
        **kwargs,
    )
    return _pack_dataset(X, y, input_locations)


def build_reservoir_opcode_logic16_dataset(
    *, reservoir_kind: str = "eca", reservoir_config: dict | None = None, **kwargs
):
    X, y, input_locations = build_dataset_programmed_logic16(
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
        **kwargs,
    )
    return _pack_dataset(X, y, input_locations)


def build_reservoir_compound_opcode_dataset(
    *, reservoir_kind: str = "eca", reservoir_config: dict | None = None, **kwargs
):
    X, y, input_locations = build_dataset_compound_opcode(
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
        **kwargs,
    )
    return _pack_dataset(X, y, input_locations)


def build_reservoir_bit_memory_dataset(
    *, reservoir_kind: str = "eca", reservoir_config: dict | None = None, **kwargs
):
    X, y, input_locations = build_dataset_output_window_only(
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
        **kwargs,
    )
    return _pack_dataset(X, y, input_locations)


def build_reservoir_serial_adder_dataset(
    *, reservoir_kind: str = "eca", reservoir_config: dict | None = None, **kwargs
):
    X, y, input_locations = build_dataset_serial_adder(
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
        **kwargs,
    )
    return _pack_dataset(X, y, input_locations)


def build_reservoir_toy_addition_dataset(
    n_bits: int,
    cin: int,
    *,
    reservoir_kind: str = "eca",
    reservoir_config: dict | None = None,
    **kwargs,
):
    X, y = build_addition_reservoir_dataset(
        n_bits,
        cin,
        reservoir_kind=reservoir_kind,
        reservoir_config=reservoir_config,
        **kwargs,
    )
    return _pack_dataset(X, y, None)


def build_direct_opcode_logic_dataset():
    X_all = []
    y_all = []
    for op_int in range(8):
        op_bits = np.array(
            [(op_int >> 2) & 1, (op_int >> 1) & 1, op_int & 1], dtype=np.int8
        )
        for a in (0, 1):
            for b in (0, 1):
                X_all.append(np.array([*op_bits, a, b], dtype=np.int8))
                y_all.append(apply_opcode_3(op_bits, a, b))
    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int8)
    return _pack_dataset(X, y, None)


def build_direct_opcode_logic16_dataset():
    X_all = []
    y_all = []
    for op_int in range(16):
        op_bits = np.array(
            [
                (op_int >> 3) & 1,
                (op_int >> 2) & 1,
                (op_int >> 1) & 1,
                op_int & 1,
            ],
            dtype=np.int8,
        )
        for a in (0, 1):
            for b in (0, 1):
                X_all.append(np.array([*op_bits, a, b], dtype=np.int8))
                y_all.append(apply_opcode_4(op_bits, a, b))
    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int8)
    return _pack_dataset(X, y, None)


def build_direct_compound_opcode_dataset():
    X_all = []
    y_all = []
    for op1_int in range(16):
        op1_bits = np.array(
            [
                (op1_int >> 3) & 1,
                (op1_int >> 2) & 1,
                (op1_int >> 1) & 1,
                op1_int & 1,
            ],
            dtype=np.int8,
        )
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
                    )
                    for c in (0, 1):
                        X_all.append(
                            np.array([*op1_bits, a, b, *op2_bits, c], dtype=np.int8)
                        )
                        y_all.append(apply_compound_opcode(op1_bits, a, b, op2_bits, c))
    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int8)
    return _pack_dataset(X, y, None)


def build_direct_bit_memory_dataset(bits: int):
    n_samples = 2**bits
    X_all = []
    y_all = []
    for i in range(n_samples):
        bits_arr = int_to_bits_lsb(i, bits)
        for bit_idx in range(bits):
            X_all.append(bits_arr.astype(np.int8))
            y_all.append(int(bits_arr[bit_idx]))
    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int8)
    return _pack_dataset(X, y, None)


def build_direct_serial_adder_dataset(bits: int, n_samples: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_all = []
    y_all = []
    for _ in range(int(n_samples)):
        a = int(rng.integers(0, 2**bits))
        b = int(rng.integers(0, 2**bits))
        a_bits = int_to_bits_lsb(a, bits)
        b_bits = int_to_bits_lsb(b, bits)
        sum_bits = int_to_bits_lsb(a + b, bits)
        for bit_idx in range(bits):
            X_all.append(np.concatenate([a_bits, b_bits]).astype(np.int8))
            y_all.append(int(sum_bits[bit_idx]))
    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int8)
    return _pack_dataset(X, y, None)


def build_direct_toy_addition_dataset(n_bits: int, cin: int = 0):
    X, y = enumerate_addition_dataset(n_bits, cin)
    return _pack_dataset(X, y, None)
