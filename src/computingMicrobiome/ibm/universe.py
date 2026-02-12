"""Deterministic \"universe\" of IBM species and resources.

This module defines a fixed, reusable universe of IBM species and metabolites
that can be shared across:

- Standalone IBM simulations (e.g. notebooks).
- IBM-backed reservoir computing tasks.

The core idea:

- We define a *large* parameter space once (e.g. 50 species, 100 resources).
- Each species can uptake multiple resources and secrete lower-energy
  resources, creating cross-feeding and trophic structure.
- The wiring is generated *once* using a fixed RNG seed and is therefore
  deterministic and stable across runs.
- Downstream code selects subsets of species by **index**; their resource
  usage (uptake + secretion) comes along automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .params import EnvParams, SpeciesParams, load_params
from .state import GridState


# Size of the global IBM universe.
N_SPECIES_UNIVERSE = 50
N_RESOURCES_UNIVERSE = 100


@dataclass(frozen=True)
class _UniverseParams:
    """Fully specified per-species and environment-like parameters for the universe."""

    maint_cost: np.ndarray
    uptake_rate: np.ndarray
    yield_energy: np.ndarray
    div_threshold: np.ndarray
    div_cost: np.ndarray
    birth_energy: np.ndarray
    uptake_list: tuple[np.ndarray, ...]
    secrete_list: tuple[np.ndarray, ...]
    feed_rate: np.ndarray


_UNIVERSE: _UniverseParams | None = None


def _build_universe() -> _UniverseParams:
    """Construct a deterministic species–resource universe.

    Design:
    - Resources split into three energy bands:
        * high:  0–9
        * mid:   10–59
        * low:   60–99
    - Species partitioned into groups:
        * high-eaters (0–19): primary uptake from high, secrete into mid.
        * mid-eaters (20–39): primary uptake from mid, secrete into low.
        * low-eaters (40–49): primary uptake from low, secrete into low
          (decomposers / recyclers).
    - Each species uptakes multiple resources (primary + secondary) and
      secretes a small set of lower-energy resources.
    - External feed is concentrated on a few high-band resources and lightly
      on some mid-band resources; low-band resources appear mostly via
      secretion.
    """

    rng = np.random.default_rng(12345)

    # Resource energy bands.
    high = np.arange(0, 10, dtype=np.int16)
    mid = np.arange(10, 60, dtype=np.int16)
    low = np.arange(60, 100, dtype=np.int16)

    maint_cost = np.zeros(N_SPECIES_UNIVERSE, dtype=np.int16)
    uptake_rate = np.zeros(N_SPECIES_UNIVERSE, dtype=np.int16)
    yield_energy = np.zeros(N_SPECIES_UNIVERSE, dtype=np.int16)
    div_threshold = np.zeros(N_SPECIES_UNIVERSE, dtype=np.int16)
    div_cost = np.zeros(N_SPECIES_UNIVERSE, dtype=np.int16)
    birth_energy = np.zeros(N_SPECIES_UNIVERSE, dtype=np.int16)

    uptake_list: list[np.ndarray] = []
    secrete_list: list[np.ndarray] = []

    for s in range(N_SPECIES_UNIVERSE):
        # Choose group based on species index.
        if s < 20:
            home_band = high
            secrete_band = mid
        elif s < 40:
            home_band = mid
            secrete_band = low
        else:
            home_band = low
            secrete_band = low

        # Primary uptake: 2–3 resources from the home band.
        n_primary = int(rng.integers(2, 4))
        primary = rng.choice(home_band, size=n_primary, replace=False)

        # Secondary uptake: 1–3 resources from the full resource set.
        all_resources = np.arange(N_RESOURCES_UNIVERSE, dtype=np.int16)
        n_secondary = int(rng.integers(1, 4))
        secondary = rng.choice(all_resources, size=n_secondary, replace=False)

        uptake = np.unique(np.concatenate([primary, secondary])).astype(np.int16)
        uptake.sort()

        # Secretion: small number of targets from the secrete band.
        # High-band species: mostly 0 or 1 secretion target to avoid
        # excessive mass amplification; others: 1–2 targets.
        if s < 20:
            # 50% chance of no secretion, otherwise exactly one target.
            if rng.random() < 0.5:
                secrete = np.empty(0, dtype=np.int16)
            else:
                secrete = rng.choice(secrete_band, size=1, replace=False).astype(np.int16)
        else:
            n_secrete = int(rng.integers(1, 3))
            n_secrete = min(n_secrete, max(1, secrete_band.size))
            secrete = rng.choice(
                secrete_band,
                size=n_secrete,
                replace=secrete_band.size < n_secrete,
            ).astype(np.int16)

        if secrete.size > 0:
            secrete = np.unique(secrete.astype(np.int16))
            secrete.sort()

        uptake_list.append(uptake)
        secrete_list.append(secrete)

        # Scalar parameters: chosen from reasonable integer ranges and later
        # clipped inside `load_params`.
        maint_cost[s] = int(rng.integers(1, 4))       # 1–3
        uptake_rate[s] = int(rng.integers(1, 4))      # 1–3 uptake attempts / tick
        yield_energy[s] = int(rng.integers(3, 7))     # 3–6 energy per successful uptake
        div_threshold[s] = int(rng.integers(18, 34))  # energy needed to divide
        div_cost[s] = int(rng.integers(8, 18))        # energy cost of division
        birth_energy[s] = int(rng.integers(10, 20))   # newborn energy

    # External feed pattern: modest feed on a few high-band resources, weaker on
    # some mid-band resources, almost none elsewhere. Combined with dilution,
    # this aims to keep total resource levels roughly bounded.
    feed_rate = np.zeros(N_RESOURCES_UNIVERSE, dtype=np.float32)

    # 3 high-energy resources with modest feed.
    high_feed = rng.choice(high, size=3, replace=False)
    feed_rate[high_feed] = rng.uniform(0.5, 1.5, size=3).astype(np.float32)

    # 5 mid-band resources with weak feed.
    mid_feed = rng.choice(mid, size=5, replace=False)
    feed_rate[mid_feed] = rng.uniform(0.05, 0.3, size=5).astype(np.float32)

    # Tiny baseline feed on a few low-band resources.
    low_feed = rng.choice(low, size=5, replace=False)
    feed_rate[low_feed] = rng.uniform(0.0, 0.05, size=5).astype(np.float32)

    return _UniverseParams(
        maint_cost=maint_cost,
        uptake_rate=uptake_rate,
        yield_energy=yield_energy,
        div_threshold=div_threshold,
        div_cost=div_cost,
        birth_energy=birth_energy,
        uptake_list=tuple(arr.copy() for arr in uptake_list),
        secrete_list=tuple(arr.copy() for arr in secrete_list),
        feed_rate=feed_rate,
    )


def _get_universe() -> _UniverseParams:
    global _UNIVERSE
    if _UNIVERSE is None:
        _UNIVERSE = _build_universe()
    return _UNIVERSE


def _normalize_species_indices(indices: Sequence[int] | None) -> list[int]:
    if indices is None:
        return list(range(N_SPECIES_UNIVERSE))
    out: list[int] = []
    for x in indices:
        i = int(x)
        if not (0 <= i < N_SPECIES_UNIVERSE):
            raise ValueError(f"species index {i} out of range [0, {N_SPECIES_UNIVERSE})")
        out.append(i)
    if not out:
        raise ValueError("at least one species index is required")
    return out


def make_ibm_config_from_species(
    species_indices: Sequence[int] | None = None,
    *,
    height: int = 16,
    width_grid: int = 32,
    Rmax: int = 255,
    Emax: int = 255,
    diff_numer: int = 1,
    diff_denom: int = 8,
    transport_shift: int = 0,
    dilution_p: float = 0.02,
    inject_scale: float = 5.0,
    basal_init: bool = True,
    basal_occupancy: float = 0.6,
    basal_energy: int = 12,
    basal_resource: int = 10,
    basal_pattern: str = "checkerboard",
    overrides: Mapping[str, object] | None = None,
) -> dict:
    """Build a `load_params`-compatible IBM config from a subset of the universe.

    Parameters
    ----------
    species_indices:
        Indices into the global universe [0, N_SPECIES_UNIVERSE). If None,
        uses all universe species.
    height, width_grid:
        Lattice dimensions.
    Rmax, Emax, diff_numer, diff_denom, transport_shift, dilution_p,
    inject_scale, basal_*:
        Environment-level parameters passed directly to `load_params`.
    overrides:
        Optional extra keys/values to merge into the config; this is useful to
        attach reservoir-backend-specific options such as `state_width_mode`
        or `input_trace_depth` that are *not* consumed by `load_params` but are
        read by `IBMReservoirBackend`.
    """
    uni = _get_universe()
    idx = _normalize_species_indices(species_indices)

    n_species = len(idx)

    # Determine the minimal set of resources that are actually relevant for the
    # selected species: any resource they uptake, secrete, or that receives
    # external feed. This is then remapped to a compact index set [0, M).
    used: set[int] = set()
    for i in idx:
        used.update(int(x) for x in uni.uptake_list[i].tolist())
        used.update(int(x) for x in uni.secrete_list[i].tolist())

    # Always include resources that receive non-zero external feed.
    feed_nonzero = np.nonzero(uni.feed_rate > 0.0)[0].tolist()
    used.update(int(x) for x in feed_nonzero)

    if not used:
        # Fallback: in the unlikely case nothing was marked as used, keep at
        # least one resource so the system is well-defined.
        used.add(0)

    res_indices = sorted(used)
    n_resources = len(res_indices)

    # Mapping from original resource indices [0, N_RESOURCES_UNIVERSE) to
    # compacted indices [0, n_resources).
    res_map = -np.ones(N_RESOURCES_UNIVERSE, dtype=np.int16)
    for new_idx, old_idx in enumerate(res_indices):
        res_map[int(old_idx)] = int(new_idx)

    # Slice feed_rate and remap per-species resource lists.
    feed_rate_compact = uni.feed_rate[res_indices]

    uptake_compact: list[list[int]] = []
    secrete_compact: list[list[int]] = []
    for i in idx:
        u_src = uni.uptake_list[i]
        s_src = uni.secrete_list[i]
        u_mapped = sorted({int(res_map[int(x)]) for x in u_src.tolist() if res_map[int(x)] >= 0})
        s_mapped = sorted({int(res_map[int(x)]) for x in s_src.tolist() if res_map[int(x)] >= 0})
        uptake_compact.append(u_mapped)
        secrete_compact.append(s_mapped)

    # Build a low per-resource basal level aligned with the feed pattern.
    if feed_rate_compact.size > 0 and float(feed_rate_compact.max()) > 0.0:
        max_basal = max(1.0, min(5.0, Rmax / 20.0))
        scaled = feed_rate_compact.astype(np.float32) / float(feed_rate_compact.max())
        basal_vec = np.rint(scaled * max_basal).astype(np.int16)
    else:
        basal_vec = np.zeros(n_resources, dtype=np.int16)

    cfg: dict[str, object] = {
        "height": int(height),
        "width_grid": int(width_grid),
        "n_species": int(n_species),
        "n_resources": int(n_resources),
        "Rmax": int(Rmax),
        "Emax": int(Emax),
        "diff_numer": int(diff_numer),
        "diff_denom": int(diff_denom),
        "transport_shift": int(transport_shift),
        "dilution_p": float(dilution_p),
        "feed_rate": feed_rate_compact.tolist(),
        "inject_scale": float(inject_scale),
        "basal_init": bool(basal_init),
        "basal_occupancy": float(basal_occupancy),
        "basal_energy": int(basal_energy),
        # Use per-resource basal levels to roughly align the initial condition
        # with the feed pattern, while keeping them globally low.
        "basal_resource": basal_vec.tolist(),
        "basal_pattern": str(basal_pattern),
        # Per-species scalar parameters.
        "maint_cost": uni.maint_cost[idx].tolist(),
        "uptake_rate": uni.uptake_rate[idx].tolist(),
        "yield_energy": uni.yield_energy[idx].tolist(),
        "div_threshold": uni.div_threshold[idx].tolist(),
        "div_cost": uni.div_cost[idx].tolist(),
        "birth_energy": uni.birth_energy[idx].tolist(),
        # Per-species resource lists.
        "uptake_list": uptake_compact,
        "secrete_list": secrete_compact,
    }

    if overrides:
        cfg.update(dict(overrides))

    return cfg


def make_env_and_species_from_species(
    species_indices: Sequence[int] | None = None,
    *,
    height: int = 16,
    width_grid: int = 32,
    overrides: Mapping[str, object] | None = None,
) -> tuple[EnvParams, SpeciesParams]:
    """Convenience wrapper returning `(EnvParams, SpeciesParams)`."""
    cfg = make_ibm_config_from_species(
        species_indices,
        height=height,
        width_grid=width_grid,
        overrides=overrides,
    )
    env, species = load_params(cfg)
    return env, species


def make_center_column_state(
    env: EnvParams,
    *,
    species_id: int = 0,
    energy_mean: float = 4.0,
    rng: np.random.Generator | None = None,
) -> GridState:
    """Initialize a grid with a single vertical column of one species.

    - Only the central column is occupied (all rows).
    - Occupied cells have energy ~ energy_mean (with small noise).
    - Resources are initialized from basal_resource_vec / basal_resource.
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = env.height, env.width_grid
    mid = W // 2

    # Occupancy: everything empty except the central column.
    occ = np.full((H, W), -1, dtype=np.int16)
    occ[:, mid] = int(species_id)

    # Energy: around energy_mean, clipped to [0, Emax].
    E = np.zeros((H, W), dtype=np.uint8)
    vals = rng.normal(loc=energy_mean, scale=1.0, size=(H,))
    vals = np.clip(vals, 0, env.Emax).astype(np.uint8)
    E[:, mid] = vals

    # Resources: use per-resource basal if available, else scalar.
    br_vec = getattr(env, "basal_resource_vec", None)
    if br_vec is not None:
        br = np.asarray(br_vec, dtype=np.uint8).reshape(env.n_resources)
        R = np.broadcast_to(
            br[:, None, None],
            (env.n_resources, H, W),
        ).copy()
    else:
        R = np.full(
            (env.n_resources, H, W),
            np.uint8(env.basal_resource),
            dtype=np.uint8,
        )

    return GridState(occ=occ, E=E, R=R)



