"""Grid state for the IBM reservoir core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridState:
    """Discrete lattice state.

    Attributes:
        occ: Occupancy map ``(H, W)`` with ``-1`` for empty, else species id.
        E: Per-cell energy ``(H, W)``.
        R: Resource tensor ``(M, H, W)``.
    """

    occ: np.ndarray
    E: np.ndarray
    R: np.ndarray


def make_zero_state(*, height: int, width_grid: int, n_resources: int) -> GridState:
    """Allocate an empty state with uint8 resources and energies."""
    occ = np.full((height, width_grid), -1, dtype=np.int16)
    E = np.zeros((height, width_grid), dtype=np.uint8)
    R = np.zeros((n_resources, height, width_grid), dtype=np.uint8)
    return GridState(occ=occ, E=E, R=R)
