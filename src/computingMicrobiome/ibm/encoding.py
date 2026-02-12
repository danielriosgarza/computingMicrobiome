"""State encoding for fixed-width reservoir readout features."""

from __future__ import annotations

import numpy as np

from .params import EnvParams
from .state import GridState


def encode_state(
    state: GridState,
    env: EnvParams,
    *,
    output_width: int,
) -> np.ndarray:
    """Encode ``(occ, E, R)`` into a 1D float32 feature vector."""
    occ = state.occ.reshape(-1)
    occ_onehot = np.zeros((env.n_species, occ.size), dtype=np.float32)
    for s in range(env.n_species):
        occ_onehot[s] = (occ == s).astype(np.float32)

    E_flat = state.E.reshape(-1).astype(np.float32) / float(max(env.Emax, 1))
    R_flat = state.R.reshape(-1).astype(np.float32) / float(max(env.Rmax, 1))
    counts = np.bincount(occ[occ >= 0], minlength=env.n_species).astype(
        np.float32
    )
    counts /= float(max(occ.size, 1))

    raw = np.concatenate([occ_onehot.reshape(-1), E_flat, R_flat, counts]).astype(
        np.float32
    )

    if output_width == raw.size:
        return raw
    if output_width < 1:
        raise ValueError("output_width must be >= 1")

    if raw.size < output_width:
        out = np.zeros(output_width, dtype=np.float32)
        out[: raw.size] = raw
        return out

    # Fold to fixed width while keeping scale stable.
    out = np.zeros(output_width, dtype=np.float32)
    idx = np.arange(raw.size, dtype=np.int64) % int(output_width)
    np.add.at(out, idx, raw)
    counts_fold = np.bincount(idx, minlength=output_width).astype(np.float32)
    out /= np.maximum(counts_fold, 1.0)
    return out
