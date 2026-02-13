"""Bit pulse injection (toxin vs popular metabolite) for IBM reservoir."""

from __future__ import annotations

import numpy as np

from .params import EnvParams, SpeciesParams
from .state import GridState


def square_mask(
    height: int, width: int, center_r: int, center_c: int, radius: int
) -> np.ndarray:
    """Boolean mask for the square region (Chebyshev distance <= radius)."""
    rr = np.arange(height)[:, None]
    cc = np.arange(width)[None, :]
    return (np.abs(rr - center_r) <= radius) & (np.abs(cc - center_c) <= radius)


def get_popular_resource_index(species: SpeciesParams) -> int:
    """First resource index in any species' popular_uptake_list (compact index)."""
    seen = set()
    for arr in species.popular_uptake_list:
        for m in arr.tolist():
            seen.add(int(m))
    if not seen:
        raise ValueError(
            "No popular_uptake_list resources found; config may omit popular metabolite."
        )
    return min(seen)


def inject_bit_into_state(
    state: GridState,
    env: EnvParams,
    species: SpeciesParams,
    bit: int,
    center_r: int,
    center_c: int,
    radius: int,
    toxin_conc: int = 200,
    popular_conc: int = 200,
) -> None:
    """Apply a clean bit pulse: clear square (occ, E, all R), then set toxin (0) or popular (1)."""
    H, W = env.height, env.width_grid
    mask = square_mask(H, W, center_r, center_c, radius)

    state.occ[mask] = -1
    state.E[mask] = 0
    state.R[:, mask] = 0

    if env.toxin_resource_index is None:
        raise ValueError("Config has no toxin_resource_index; required for bit-0 pulse.")

    if bit == 0:
        state.R[env.toxin_resource_index, mask] = np.clip(
            toxin_conc, 0, env.Rmax
        ).astype(np.uint8)
    else:
        pop_idx = get_popular_resource_index(species)
        state.R[pop_idx, mask] = np.clip(popular_conc, 0, env.Rmax).astype(np.uint8)
