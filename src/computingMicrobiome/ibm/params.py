"""Configuration parsing for the IBM reservoir core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SpeciesParams:
    """Per-species discrete dynamics parameters."""

    maint_cost: np.ndarray
    uptake_rate: np.ndarray
    uptake_list: tuple[np.ndarray, ...]
    popular_uptake_list: tuple[np.ndarray, ...]
    secondary_uptake: np.ndarray
    secrete_list: tuple[np.ndarray, ...]
    yield_energy: np.ndarray
    div_threshold: np.ndarray
    div_cost: np.ndarray
    birth_energy: np.ndarray
    # Per-species max stored energy (~5× div_cost); limits saturation.
    energy_capacity: np.ndarray


@dataclass(frozen=True)
class EnvParams:
    """Environment/lattice parameters."""

    height: int
    width_grid: int
    n_species: int
    n_resources: int
    Rmax: int
    Emax: int
    diff_numer: int
    diff_denom: int
    transport_shift: int
    dilution_p: float
    feed_rate: np.ndarray
    inject_scale: float
    channel_to_resource: np.ndarray | None
    allow_invasion: bool
    invasion_energy_margin: int
    basal_init: bool
    basal_occupancy: float
    basal_energy: int
    basal_resource: int
    # Optional per-resource basal levels (length n_resources). When provided,
    # this should be preferred over the scalar `basal_resource` for initializing
    # resource fields. Older configs that only specify a scalar remain valid.
    basal_resource_vec: np.ndarray | None
    basal_pattern: str


def _read_species_scalar(
    cfg: Mapping[str, Any],
    key: str,
    n_species: int,
    default: int,
) -> np.ndarray:
    raw = cfg.get(key, default)
    if np.isscalar(raw):
        return np.full(n_species, int(raw), dtype=np.int16)
    arr = np.asarray(raw, dtype=np.int16).reshape(-1)
    if arr.size != n_species:
        raise ValueError(f"{key} must be scalar or length n_species")
    return arr.copy()


def _coerce_resource_list(v: Any, n_resources: int, *, key: str) -> np.ndarray:
    arr = np.asarray(v, dtype=np.int16).reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype=np.int16)
    if np.any(arr < 0) or np.any(arr >= n_resources):
        raise ValueError(f"{key} values must be in [0, n_resources)")
    return arr


def _dedupe_resource_list(arr: np.ndarray) -> np.ndarray:
    if arr.size < 2:
        return arr.copy()
    out: list[int] = []
    seen: set[int] = set()
    for x in arr.tolist():
        val = int(x)
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return np.asarray(out, dtype=np.int16)


def _read_species_optional_resource_scalar(
    cfg: Mapping[str, Any],
    key: str,
    n_species: int,
    n_resources: int,
) -> np.ndarray:
    raw = cfg.get(key)
    if raw is None:
        return np.full(n_species, -1, dtype=np.int16)
    if np.isscalar(raw):
        out = np.full(n_species, int(raw), dtype=np.int16)
    else:
        out = np.asarray(raw, dtype=np.int16).reshape(-1)
        if out.size != n_species:
            raise ValueError(f"{key} must be scalar or length n_species")
    if np.any(out < -1) or np.any(out >= n_resources):
        raise ValueError(f"{key} values must be -1 or in [0, n_resources)")
    return out.copy()


def _read_species_resource_lists(
    cfg: Mapping[str, Any],
    key: str,
    n_species: int,
    n_resources: int,
    *,
    default_all_resources: bool,
) -> list[np.ndarray]:
    default_single = (
        np.arange(n_resources, dtype=np.int16)
        if default_all_resources
        else np.empty(0, dtype=np.int16)
    )

    raw = cfg.get(key)
    if raw is None:
        return [default_single.copy() for _ in range(n_species)]

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")

    if len(raw) == 0:
        return [np.empty(0, dtype=np.int16) for _ in range(n_species)]

    first = raw[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        if len(raw) != n_species:
            raise ValueError(f"{key} nested form must have length n_species")
        return [
            _coerce_resource_list(raw[s], n_resources, key=key) for s in range(n_species)
        ]

    single = _coerce_resource_list(raw, n_resources, key=key)
    return [single.copy() for _ in range(n_species)]


def load_params(config: Mapping[str, Any] | None) -> tuple[EnvParams, SpeciesParams]:
    """Parse backend config into immutable parameter dataclasses."""
    cfg = dict(config or {})

    height = int(cfg.get("height", 8))
    width_grid = int(cfg.get("width_grid", 8))
    n_species = int(cfg.get("n_species", 2))
    n_resources = int(cfg.get("n_resources", 2))
    if height < 1 or width_grid < 1:
        raise ValueError("IBM config requires height >= 1 and width_grid >= 1")
    if n_species < 1 or n_resources < 1:
        raise ValueError("IBM config requires n_species >= 1 and n_resources >= 1")

    Rmax = int(cfg.get("Rmax", 255))
    Emax = int(cfg.get("Emax", 255))
    if Rmax < 1 or Emax < 1 or Rmax > 255 or Emax > 255:
        raise ValueError("IBM config requires 1 <= Rmax,Emax <= 255")

    diff_numer = int(cfg.get("diff_numer", 1))
    diff_denom = int(cfg.get("diff_denom", 8))
    if diff_numer < 0 or diff_denom < 1:
        raise ValueError("IBM config requires diff_numer >= 0 and diff_denom >= 1")
    transport_shift = int(cfg.get("transport_shift", 0))

    dilution_p = float(cfg.get("dilution_p", 0.01))
    if not (0.0 <= dilution_p <= 1.0):
        raise ValueError("IBM config requires 0 <= dilution_p <= 1")

    inject_scale = float(cfg.get("inject_scale", 5.0))

    raw_feed = cfg.get("feed_rate", 0.0)
    if np.isscalar(raw_feed):
        feed_rate = np.full(n_resources, float(raw_feed), dtype=np.float32)
    else:
        feed_rate = np.asarray(raw_feed, dtype=np.float32).reshape(-1)
        if feed_rate.size != n_resources:
            raise ValueError("feed_rate must be scalar or length n_resources")
    feed_rate = np.clip(feed_rate, 0.0, float(Rmax))

    raw_map = cfg.get("channel_to_resource")
    if raw_map is None:
        channel_to_resource = None
    else:
        channel_to_resource = np.asarray(raw_map, dtype=np.int16).reshape(-1)
        if channel_to_resource.size < 1:
            raise ValueError("channel_to_resource cannot be empty")
        if np.any(channel_to_resource < 0) or np.any(channel_to_resource >= n_resources):
            raise ValueError("channel_to_resource values must be in [0, n_resources)")

    allow_invasion = bool(cfg.get("allow_invasion", False))
    invasion_energy_margin = int(cfg.get("invasion_energy_margin", 0))
    if invasion_energy_margin < 0:
        raise ValueError("invasion_energy_margin must be >= 0")

    basal_init = bool(cfg.get("basal_init", False))
    basal_occupancy = float(cfg.get("basal_occupancy", 1.0))
    if not (0.0 <= basal_occupancy <= 1.0):
        raise ValueError("basal_occupancy must be in [0, 1]")
    basal_energy = int(cfg.get("basal_energy", 12))

    # Support both scalar and per-resource basal_resource specifications.
    raw_basal = cfg.get("basal_resource", 2)
    if np.isscalar(raw_basal):
        basal_resource = int(raw_basal)
        basal_resource_vec = np.full(n_resources, basal_resource, dtype=np.uint8)
    else:
        arr = np.asarray(raw_basal, dtype=np.int16).reshape(-1)
        if arr.size != n_resources:
            raise ValueError("basal_resource sequence must have length n_resources")
        basal_resource_vec = np.clip(arr, 0, Rmax).astype(np.uint8)
        # Use the mean as a representative scalar for backwards-compatible
        # consumers that still read `basal_resource`.
        basal_resource = int(basal_resource_vec.mean())
    basal_pattern = str(cfg.get("basal_pattern", "checkerboard")).strip().lower()
    if basal_pattern not in {"checkerboard", "stripes"}:
        raise ValueError("basal_pattern must be one of {'checkerboard', 'stripes'}")

    maint_cost = _read_species_scalar(cfg, "maint_cost", n_species, default=1)
    uptake_rate = _read_species_scalar(cfg, "uptake_rate", n_species, default=1)
    yield_energy = _read_species_scalar(cfg, "yield_energy", n_species, default=4)
    div_threshold = _read_species_scalar(cfg, "div_threshold", n_species, default=32)
    div_cost = _read_species_scalar(cfg, "div_cost", n_species, default=16)
    birth_energy = _read_species_scalar(cfg, "birth_energy", n_species, default=16)

    uptake_list = _read_species_resource_lists(
        cfg,
        "uptake_list",
        n_species,
        n_resources,
        default_all_resources=True,
    )
    secrete_list = _read_species_resource_lists(
        cfg,
        "secrete_list",
        n_species,
        n_resources,
        default_all_resources=False,
    )

    raw_popular = cfg.get("popular_uptake_list")
    if raw_popular is None:
        popular_uptake_list = [np.empty(0, dtype=np.int16) for _ in range(n_species)]
    else:
        popular_global = _dedupe_resource_list(
            _coerce_resource_list(
                raw_popular,
                n_resources,
                key="popular_uptake_list",
            )
        )
        popular_uptake_list = []
        for s in range(n_species):
            allowed = set(int(x) for x in uptake_list[s].tolist())
            popular_s = np.asarray(
                [int(m) for m in popular_global.tolist() if int(m) in allowed],
                dtype=np.int16,
            )
            popular_uptake_list.append(popular_s)

    secondary_uptake = _read_species_optional_resource_scalar(
        cfg,
        "secondary_uptake",
        n_species,
        n_resources,
    )

    species_cfg = cfg.get("species") or cfg.get("species_params")
    if species_cfg is not None:
        if not isinstance(species_cfg, Sequence) or len(species_cfg) != n_species:
            raise ValueError("species/species_params must have length n_species")
        for s, raw_entry in enumerate(species_cfg):
            entry = dict(raw_entry or {})
            if "maint_cost" in entry:
                maint_cost[s] = int(entry["maint_cost"])
            if "uptake_rate" in entry:
                uptake_rate[s] = int(entry["uptake_rate"])
            if "yield_energy" in entry:
                yield_energy[s] = int(entry["yield_energy"])
            if "div_threshold" in entry:
                div_threshold[s] = int(entry["div_threshold"])
            if "div_cost" in entry:
                div_cost[s] = int(entry["div_cost"])
            if "birth_energy" in entry:
                birth_energy[s] = int(entry["birth_energy"])
            if "energy_capacity" in entry:
                energy_capacity[s] = int(np.clip(entry["energy_capacity"], 1, Emax))
            if "uptake_list" in entry:
                uptake_list[s] = _coerce_resource_list(
                    entry["uptake_list"],
                    n_resources,
                    key="uptake_list",
                )
            if "secrete_list" in entry:
                secrete_list[s] = _coerce_resource_list(
                    entry["secrete_list"],
                    n_resources,
                    key="secrete_list",
                )
            if "popular_uptake_list" in entry:
                popular_uptake_list[s] = _dedupe_resource_list(
                    _coerce_resource_list(
                        entry["popular_uptake_list"],
                        n_resources,
                        key="popular_uptake_list",
                    )
                )
            if "secondary_uptake" in entry:
                secondary_uptake[s] = int(entry["secondary_uptake"])

    maint_cost = np.clip(maint_cost, 0, Emax).astype(np.uint8)
    uptake_rate = np.clip(uptake_rate, 0, 255).astype(np.int16)
    yield_energy = np.clip(yield_energy, 0, Emax).astype(np.uint8)
    div_threshold = np.clip(div_threshold, 0, Emax).astype(np.uint8)
    div_cost = np.clip(div_cost, 0, Emax).astype(np.uint8)
    birth_energy = np.clip(birth_energy, 0, Emax).astype(np.uint8)

    # Per-species energy capacity: default ~3× div_cost to limit saturation.
    raw_cap = cfg.get("energy_capacity")
    if raw_cap is None:
        energy_capacity = np.clip(3 * div_cost.astype(np.int32), 1, Emax).astype(
            np.uint8
        )
    else:
        energy_capacity = _read_species_scalar(
            cfg, "energy_capacity", n_species, default=3 * int(div_cost[0])
        )
        energy_capacity = np.clip(energy_capacity.astype(np.int32), 1, Emax).astype(
            np.uint8
        )

    if np.any(secondary_uptake < -1) or np.any(secondary_uptake >= n_resources):
        raise ValueError("secondary_uptake values must be -1 or in [0, n_resources)")

    for s in range(n_species):
        allowed = set(int(x) for x in uptake_list[s].tolist())
        pop = popular_uptake_list[s]
        if pop.size > 0:
            invalid_pop = [int(x) for x in pop.tolist() if int(x) not in allowed]
            if invalid_pop:
                raise ValueError(
                    f"popular_uptake_list for species {s} must be subset of uptake_list"
                )
        sec = int(secondary_uptake[s])
        if sec >= 0 and sec not in allowed:
            raise ValueError(
                f"secondary_uptake for species {s} must be in uptake_list or -1"
            )

    env = EnvParams(
        height=height,
        width_grid=width_grid,
        n_species=n_species,
        n_resources=n_resources,
        Rmax=Rmax,
        Emax=Emax,
        diff_numer=diff_numer,
        diff_denom=diff_denom,
        transport_shift=transport_shift,
        dilution_p=dilution_p,
        feed_rate=feed_rate,
        inject_scale=inject_scale,
        channel_to_resource=channel_to_resource,
        allow_invasion=allow_invasion,
        invasion_energy_margin=int(np.clip(invasion_energy_margin, 0, Emax)),
        basal_init=basal_init,
        basal_occupancy=basal_occupancy,
        basal_energy=int(np.clip(basal_energy, 0, Emax)),
        basal_resource=int(np.clip(basal_resource, 0, Rmax)),
        basal_resource_vec=basal_resource_vec,
        basal_pattern=basal_pattern,
    )

    species = SpeciesParams(
        maint_cost=maint_cost,
        uptake_rate=uptake_rate,
        uptake_list=tuple(arr.copy() for arr in uptake_list),
        popular_uptake_list=tuple(arr.copy() for arr in popular_uptake_list),
        secondary_uptake=secondary_uptake.astype(np.int16, copy=True),
        secrete_list=tuple(arr.copy() for arr in secrete_list),
        yield_energy=yield_energy,
        div_threshold=div_threshold,
        div_cost=div_cost,
        birth_energy=birth_energy,
        energy_capacity=energy_capacity,
    )
    return env, species
