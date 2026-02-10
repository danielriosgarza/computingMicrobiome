"""Representation adapters for the k-bit memory benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..base import RepresentationProtocol
from ...benchmarks.k_bit_memory_bm import run_episode_record, true_bits_from_episode_outputs
from ...utils import create_input_locations


def _episode_to_output_window_features(
    bits_arr: np.ndarray,
    rule_number: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    rng: np.random.Generator,
    input_locations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared helper wrapping the existing memory benchmark episode logic.

    This function is used for BOTH support and challenge transformations to
    ensure train/eval parity and to remain aligned with the existing
    reservoir/SVM memory benchmark pipeline.
    """
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
        x0_mode="zeros",
    )
    X_out = ep["X_tick"][-bits_arr.size :]
    true_bits = true_bits_from_episode_outputs(ep["outputs_tick"], bits_arr.size)
    return X_out, true_bits


@dataclass
class DirectMemoryRepresentation(RepresentationProtocol):
    """Direct bit-vector representation without reservoir dynamics.

    This adapter converts raw bit vectors of shape ``(n_samples, bits)``
    into per-bit query rows ``X`` and labels identical to those used in
    the reservoir condition.

    For each sample and queried bit index ``k``, we build one row:
      X_row = [bits_vector, one_hot_query(k)]
      y_row = bits_vector[k]
    """

    bits: int

    def transform(
        self, raw_x: np.ndarray, rng: np.random.Generator | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_x = np.asarray(raw_x, dtype=np.int8)
        if raw_x.ndim != 2 or raw_x.shape[1] != self.bits:
            raise ValueError("raw_x must have shape (n_samples, bits)")
        n_samples = raw_x.shape[0]

        X_rows = []
        y_rows = []
        eye = np.eye(self.bits, dtype=np.int8)

        for bits_arr in raw_x:
            for k in range(self.bits):
                feat = np.concatenate([bits_arr, eye[k]], axis=0)
                X_rows.append(feat)
                y_rows.append(int(bits_arr[k]))

        X = np.asarray(X_rows, dtype=float).reshape(n_samples * self.bits, 2 * self.bits)
        y = np.asarray(y_rows, dtype=np.int8).reshape(n_samples * self.bits)
        return X, y


@dataclass
class ReservoirMemoryRepresentation(RepresentationProtocol):
    """Reservoir-based representation using the existing memory benchmark code."""

    bits: int
    rule_number: int
    width: int
    boundary: str
    recurrence: int
    itr: int
    d_period: int
    seed_input_locations: int = 0

    def __post_init__(self) -> None:
        # Cache input locations per experiment unless caller wants to manage them.
        rng = np.random.default_rng(self.seed_input_locations)
        self._input_locations = create_input_locations(
            self.width, self.recurrence, 4, rng
        )

    def transform(
        self, raw_x: np.ndarray, rng: np.random.Generator | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform bit-vector episodes into reservoir features and labels.

        ``raw_x`` is expected to have shape ``(n_samples, bits)``; each
        row is treated as the bit vector for a full memory episode.
        """
        raw_x = np.asarray(raw_x, dtype=np.int8)
        if raw_x.ndim != 2 or raw_x.shape[1] != self.bits:
            raise ValueError("raw_x must have shape (n_samples, bits)")
        if rng is None:
            rng = np.random.default_rng()

        X_all = []
        y_all = []
        for bits_arr in raw_x:
            X_out, y_bits = _episode_to_output_window_features(
                bits_arr=bits_arr,
                rule_number=self.rule_number,
                width=self.width,
                boundary=self.boundary,
                recurrence=self.recurrence,
                itr=self.itr,
                d_period=self.d_period,
                rng=rng,
                input_locations=self._input_locations,
            )
            X_all.append(X_out)
            y_all.append(y_bits)

        X = np.vstack(X_all).astype(float)
        y = np.concatenate(y_all).astype(np.int8)
        return X, y


__all__ = [
    "DirectMemoryRepresentation",
    "ReservoirMemoryRepresentation",
]

