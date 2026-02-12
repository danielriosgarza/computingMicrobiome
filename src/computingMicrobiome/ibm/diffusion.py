"""Deterministic conservative diffusion for discrete resources."""

from __future__ import annotations

import numpy as np

from .params import EnvParams
from .state import GridState


def diffuse_resources(state: GridState, env: EnvParams) -> None:
    """Diffuse resources with periodic boundaries.

    Uses integer transport with deterministic remainder routing. Mass is
    conserved before clipping.
    """
    if env.diff_numer <= 0:
        return

    R = state.R.astype(np.int32, copy=False)
    moves = (R * int(env.diff_numer)) // int(env.diff_denom)
    share = moves // 4
    rem = moves % 4

    # Deterministic remainder split in the order up, right, down, left.
    to_up = share + (rem >= 1)
    to_right = share + (rem >= 2)
    to_down = share + (rem >= 3)
    to_left = share

    out = to_up + to_right + to_down + to_left
    incoming = (
        np.roll(to_up, -1, axis=1)
        + np.roll(to_down, 1, axis=1)
        + np.roll(to_left, -1, axis=2)
        + np.roll(to_right, 1, axis=2)
    )
    R_next = R - out + incoming
    if env.transport_shift != 0:
        R_next = np.roll(R_next, shift=int(env.transport_shift), axis=2)
    state.R = np.clip(R_next, 0, env.Rmax).astype(np.uint8)
