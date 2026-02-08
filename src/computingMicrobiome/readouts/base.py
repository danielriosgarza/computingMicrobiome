"""Shared readout interfaces and helpers."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Readout(Protocol):
    """Minimal interface for readout models used in benchmarks."""

    def fit(self, X: np.ndarray, y: np.ndarray): ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def decision_function(self, X: np.ndarray) -> np.ndarray: ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...


def coerce_xy_binary(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce X/y to 2D float and 1D {0,1} labels."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    y_arr = np.asarray(y).reshape(-1)
    unique = set(np.unique(y_arr).tolist())
    if unique <= {0, 1}:
        y_bin = y_arr.astype(float)
    elif unique <= {-1, 1}:
        y_bin = ((y_arr + 1) / 2).astype(float)
    else:
        raise ValueError("Binary readouts expect labels in {0,1} or {-1,1}.")

    return X_arr, y_bin
