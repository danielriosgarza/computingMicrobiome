"""Synchronous reproduction with conflict resolution."""

from __future__ import annotations

import numpy as np

from .params import EnvParams, SpeciesParams
from .state import GridState


def apply_reproduction(
    state: GridState,
    species: SpeciesParams,
    env: EnvParams,
    rng: np.random.Generator,
) -> None:
    """Apply one synchronous birth phase with target conflict resolution."""
    occ = state.occ
    H, W = occ.shape
    E_work = state.E.astype(np.int32, copy=True)

    dr = np.array([-1, 0, 1, 0], dtype=np.int16)
    dc = np.array([0, 1, 0, -1], dtype=np.int16)

    tgt_all: list[np.ndarray] = []
    par_all: list[np.ndarray] = []
    sp_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []

    for s in range(env.n_species):
        eligible = (occ == s) & (E_work >= int(species.div_threshold[s]))
        if not np.any(eligible):
            continue

        rr, cc = np.where(eligible)
        n = rr.size
        dirs = rng.integers(0, 4, size=n, dtype=np.int16)
        tr = (rr + dr[dirs]) % H
        tc = (cc + dc[dirs]) % W

        target_occ = occ[tr, tc]
        can_birth = target_occ == -1
        if env.allow_invasion:
            # Optional invasion mode: permit offspring to replace an occupied
            # neighbor of a different species if parent energy is sufficiently
            # higher than target energy.
            can_invade = (
                (target_occ >= 0)
                & (target_occ != s)
                & (
                    E_work[rr, cc]
                    >= (E_work[tr, tc] + int(env.invasion_energy_margin))
                )
            )
            can_birth = can_birth | can_invade
        if not np.any(can_birth):
            continue

        rr = rr[can_birth]
        cc = cc[can_birth]
        tr = tr[can_birth]
        tc = tc[can_birth]

        rand16 = rng.integers(0, 1 << 16, size=rr.size, dtype=np.int64)
        score = (E_work[rr, cc].astype(np.int64) << 16) + rand16

        tgt_all.append((tr * W + tc).astype(np.int64))
        par_all.append((rr * W + cc).astype(np.int64))
        sp_all.append(np.full(rr.size, s, dtype=np.int16))
        score_all.append(score)

    if not tgt_all:
        cap = np.where(occ >= 0, species.energy_capacity[occ], 0)
        state.E = np.minimum(np.maximum(E_work, 0), cap).astype(np.uint8)
        return

    tgt = np.concatenate(tgt_all)
    par = np.concatenate(par_all)
    sp = np.concatenate(sp_all)
    score = np.concatenate(score_all)

    # Sort by target ascending, then by score descending.
    order = np.lexsort((-score, tgt))
    tgt = tgt[order]
    par = par[order]
    sp = sp[order]

    keep = np.ones(tgt.size, dtype=bool)
    keep[1:] = tgt[1:] != tgt[:-1]
    tgt = tgt[keep]
    par = par[keep]
    sp = sp[keep]

    occ_flat = occ.reshape(-1)
    E_flat = E_work.reshape(-1)

    parent_cost = np.zeros_like(E_flat, dtype=np.int32)
    np.add.at(parent_cost, par, species.div_cost[sp].astype(np.int32))
    E_flat = np.maximum(E_flat - parent_cost, 0)

    occ_flat[tgt] = sp.astype(np.int16)
    E_flat[tgt] = species.birth_energy[sp].astype(np.int32)
    E_flat[occ_flat < 0] = 0

    cap_flat = np.where(occ_flat >= 0, species.energy_capacity[occ_flat], 0)
    E_flat = np.minimum(np.maximum(E_flat, 0), cap_flat.astype(np.int32))
    state.E = E_flat.reshape(H, W).astype(np.uint8)
