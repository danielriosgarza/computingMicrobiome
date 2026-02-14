"""IBM pulse backend.

This backend now reuses `IBMReservoirBackend` so reset/state-width/trace
behavior is identical to the standard IBM backend. The only difference is
default injection mode (`pulse_bit`).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..ibm.injection import PULSE_BIT
from .ibm_backend import IBMReservoirBackend


class IBMReservoirBackendPulse(IBMReservoirBackend):
    """IBM backend with `pulse_bit` as the default injection mode."""

    def __init__(self, *, config: Mapping[str, Any] | None):
        cfg = dict(config or {})
        cfg.setdefault("inject_mode", PULSE_BIT)
        super().__init__(config=cfg)

    def get_occupancy(self) -> np.ndarray:
        """Current occupancy grid (height, width_grid); -1 = empty."""
        return self._state.occ.copy()
