"""Dilution/washout dynamics for discrete resources."""

from __future__ import annotations

import numpy as np

from .params import EnvParams
from .state import GridState


def apply_dilution(state: GridState, env: EnvParams, rng: np.random.Generator) -> None:
    """Apply stochastic washout and chemostat-coupled feed inflow.

    Inflow is tied to the dilution rate: when ``dilution_p == 0``, no feed is
    injected. ``env.feed_rate`` is treated as an inflow-medium concentration in
    ``[0, Rmax]`` and converted to stochastic molecule counts per cell.
    """
    if env.dilution_p > 0.0:
        removed = rng.binomial(state.R.astype(np.int64), env.dilution_p).astype(np.int32)
        R_next = state.R.astype(np.int32) - removed
    else:
        R_next = state.R.astype(np.int32, copy=True)

    if env.dilution_p > 0.0 and np.any(env.feed_rate > 0.0):
        p_in = (
            float(env.dilution_p)
            * np.clip(
                env.feed_rate.astype(np.float64) / float(max(env.Rmax, 1)),
                0.0,
                1.0,
            )
        )
        p_grid = np.broadcast_to(p_in[:, None, None], state.R.shape)
        feed = rng.binomial(int(env.Rmax), p_grid).astype(np.int32)
        R_next = R_next + feed

    state.R = np.clip(R_next, 0, env.Rmax).astype(np.uint8)
