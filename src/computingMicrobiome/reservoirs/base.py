"""Reservoir backend protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ReservoirBackend(Protocol):
    """Minimal protocol implemented by reservoir backends."""

    width: int

    def reset(self, rng: np.random.Generator, x0_mode: str = "zeros") -> None:
        """Reset internal state."""

    def inject(
        self,
        input_values: np.ndarray,
        input_locations: np.ndarray,
        channel_idx: np.ndarray,
    ) -> None:
        """Inject one input tick into the reservoir state."""

    def step(self, rng: np.random.Generator) -> None:
        """Advance one simulation step."""

    def get_state(self) -> np.ndarray:
        """Return current 1D state vector of length ``width``."""
