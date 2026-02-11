"""ECA-backed reservoir implementation."""

from __future__ import annotations

import numpy as np

from ..eca import eca_rule_lkt, eca_step


class ECAReservoirBackend:
    """Reservoir backend that wraps elementary cellular automata dynamics."""

    def __init__(self, *, rule_number: int, width: int, boundary: str):
        self.rule_number = int(rule_number)
        self.width = int(width)
        self.boundary = str(boundary)
        self._rule = eca_rule_lkt(self.rule_number)
        self._x = np.zeros(self.width, dtype=np.int8)

    def reset(self, rng: np.random.Generator, x0_mode: str = "zeros") -> None:
        if x0_mode == "zeros":
            self._x = np.zeros(self.width, dtype=np.int8)
            return
        if x0_mode == "random":
            self._x = rng.integers(0, 2, size=self.width, dtype=np.int8)
            return
        raise ValueError("x0_mode must be 'zeros' or 'random'")

    def inject(
        self,
        input_values: np.ndarray,
        input_locations: np.ndarray,
        channel_idx: np.ndarray,
    ) -> None:
        self._x[input_locations] ^= input_values[channel_idx].astype(np.int8)

    def step(self, rng: np.random.Generator) -> None:
        self._x = eca_step(self._x, self._rule, self.boundary, rng=rng)

    def get_state(self) -> np.ndarray:
        return self._x.copy()
