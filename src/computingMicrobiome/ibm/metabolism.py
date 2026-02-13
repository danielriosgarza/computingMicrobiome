"""Energy maintenance and uptake dynamics."""

from __future__ import annotations

import numpy as np

from .params import EnvParams, SpeciesParams
from .state import GridState


def apply_maintenance(state: GridState, species: SpeciesParams, env: EnvParams) -> None:
    """Subtract maintenance cost, remove starved individuals, and apply toxin death."""
    occ = state.occ
    occupied = occ >= 0
    E_next = state.E.astype(np.int32, copy=True)
    if np.any(occupied):
        costs = np.zeros_like(E_next, dtype=np.int32)
        costs[occupied] = species.maint_cost[occ[occupied]].astype(np.int32)
        E_next = np.maximum(E_next - costs, 0)
        dead = occupied & (E_next == 0)
        occ[dead] = -1

    # Toxin death: if toxin concentration at cell > species tolerance, cell dies.
    if env.toxin_resource_index is not None and np.any(occupied):
        toxin_at_cell = state.R[env.toxin_resource_index]
        over_tolerance = np.zeros_like(occ, dtype=bool)
        over_tolerance[occupied] = (
            toxin_at_cell[occupied].astype(np.int32)
            > species.toxin_tolerance[occ[occupied]].astype(np.int32)
        )
        occ[over_tolerance] = -1

    E_next[occ < 0] = 0
    cap = np.where(occ >= 0, species.energy_capacity[occ], 0)
    state.E = np.minimum(np.maximum(E_next, 0), cap).astype(np.uint8)


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

        popular_resources = species.popular_uptake_list[s]
        secondary_resource = int(species.secondary_uptake[s])
        if popular_resources.size > 0:
            # Species-level preference policy:
            # 1) shared popular metabolites first,
            # 2) species-specific secondary fallback if popular are unavailable.
            pref_order: list[int] = []
            for m in popular_resources.tolist():
                mi = int(m)
                if mi not in pref_order:
                    pref_order.append(mi)
            if secondary_resource >= 0 and secondary_resource not in pref_order:
                pref_order.append(secondary_resource)
            uptake_order = np.asarray(pref_order, dtype=np.int16)
            if uptake_order.size == 0:
                continue
        else:
            uptake_order = uptake_resources

        secrete_resources = species.secrete_list[s]
        gain = int(species.yield_energy[s])

        for _ in range(rate):
            chosen = np.full(members.shape, -1, dtype=np.int16)
            waiting = members.copy()

            # First available resource in preference order.
            for m in uptake_order.tolist():
                can_take = waiting & (R_work[m] > 0)
                if np.any(can_take):
                    chosen[can_take] = m
                    waiting[can_take] = False
                if not np.any(waiting):
                    break

            consumed = chosen >= 0
            if not np.any(consumed):
                continue

            for m in uptake_order.tolist():
                hit = chosen == m
                if np.any(hit):
                    R_work[m, hit] -= 1

            if gain > 0:
                E_work[consumed] += gain

            for q in secrete_resources.tolist():
                R_work[q, consumed] += 1

    E_work[occ < 0] = 0
    cap = np.where(occ >= 0, species.energy_capacity[occ], 0)
    state.E = np.minimum(np.maximum(E_work, 0), cap).astype(np.uint8)
    state.R = np.clip(R_work, 0, env.Rmax).astype(np.uint8)
