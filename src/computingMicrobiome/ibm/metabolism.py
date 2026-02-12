"""Energy maintenance and uptake dynamics."""

from __future__ import annotations

import numpy as np

from .params import EnvParams, SpeciesParams
from .state import GridState


def apply_maintenance(state: GridState, species: SpeciesParams, env: EnvParams) -> None:
    """Subtract maintenance cost and remove starved individuals."""
    occ = state.occ
    occupied = occ >= 0
    E_next = state.E.astype(np.int32, copy=True)
    if np.any(occupied):
        costs = np.zeros_like(E_next, dtype=np.int32)
        costs[occupied] = species.maint_cost[occ[occupied]].astype(np.int32)
        E_next = np.maximum(E_next - costs, 0)
        dead = occupied & (E_next == 0)
        occ[dead] = -1

    E_next[occ < 0] = 0
    state.E = np.clip(E_next, 0, env.Emax).astype(np.uint8)


def apply_uptake(state: GridState, species: SpeciesParams, env: EnvParams) -> None:
    """Discrete resource uptake and energy/secretion updates."""
    occ = state.occ
    R_work = state.R.astype(np.int32, copy=True)
    E_work = state.E.astype(np.int32, copy=True)

    for s in range(env.n_species):
        members = occ == s
        if not np.any(members):
            continue

        rate = int(species.uptake_rate[s])
        if rate <= 0:
            continue

        uptake_resources = species.uptake_list[s]
        if uptake_resources.size == 0:
            continue

        secrete_resources = species.secrete_list[s]
        gain = int(species.yield_energy[s])

        for _ in range(rate):
            chosen = np.full(members.shape, -1, dtype=np.int16)
            waiting = members.copy()

            # First available resource in uptake_list order.
            for m in uptake_resources.tolist():
                can_take = waiting & (R_work[m] > 0)
                if np.any(can_take):
                    chosen[can_take] = m
                    waiting[can_take] = False
                if not np.any(waiting):
                    break

            consumed = chosen >= 0
            if not np.any(consumed):
                continue

            for m in uptake_resources.tolist():
                hit = chosen == m
                if np.any(hit):
                    R_work[m, hit] -= 1

            if gain > 0:
                E_work[consumed] += gain

            for q in secrete_resources.tolist():
                R_work[q, consumed] += 1

    E_work[occ < 0] = 0
    state.E = np.clip(E_work, 0, env.Emax).astype(np.uint8)
    state.R = np.clip(R_work, 0, env.Rmax).astype(np.uint8)
