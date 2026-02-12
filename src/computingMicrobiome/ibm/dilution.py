"""Dilution/washout dynamics for discrete resources."""

from __future__ import annotations

import numpy as np

from .params import EnvParams
from .state import GridState


def apply_dilution(state: GridState, env: EnvParams, rng: np.random.Generator) -> None:
    """Apply stochastic washout and optional feed inflow."""
    if env.dilution_p > 0.0:
        removed = rng.binomial(state.R.astype(np.int64), env.dilution_p).astype(np.int32)
        R_next = state.R.astype(np.int32) - removed
    else:
        R_next = state.R.astype(np.int32, copy=True)

    if np.any(env.feed_rate > 0.0):
        feed = np.rint(env.feed_rate).astype(np.int32)[:, None, None]
        R_next = R_next + feed

    state.R = np.clip(R_next, 0, env.Rmax).astype(np.uint8)
