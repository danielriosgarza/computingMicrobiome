"""CRM-backed reservoir implementation.

Improvements over the original MVP:

1. **Dual injection** (``inject_mode="both"``): input signal is injected
   into *both* species and resources simultaneously, doubling the
   coupling strength (analogous to ECA's direct XOR injection).

2. **Monod saturation kinetics** (``half_saturation > 0``): species
   growth saturates at high resource concentration via
   ``R / (K + R)``, producing richer nonlinear dynamics closer to
   ECA-class computational power.

3. **Per-channel state normalisation** (``normalize_state=True``):
   each species/resource field is z-scored before readout, giving
   the linear SVM features on a comparable scale.

4. **Cross-feature enrichment** (``cross_features=True``): element-wise
   species × resource products are appended to the state vector,
   exposing the nonlinear interaction structure to the readout.

All four options are **backward-compatible** and default to the
original behaviour when omitted from the config dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _laplacian_periodic(arr: np.ndarray) -> np.ndarray:
    return (
        np.roll(arr, 1, axis=-2)
        + np.roll(arr, -1, axis=-2)
        + np.roll(arr, 1, axis=-1)
        + np.roll(arr, -1, axis=-1)
        - 4.0 * arr
    )


@dataclass(frozen=True)
class CRMProjectionConfig:
    kind: str
    output_width: int | None
    seed: int
    scale: float


class CRMReservoirBackend:
    """Patch/grid CRM reservoir with deterministic state projection."""

    def __init__(self, *, config: Mapping[str, Any] | None):
        cfg = dict(config or {})

        self.height = int(cfg.get("height", 8))
        self.width_grid = int(cfg.get("width_grid", 8))
        self.n_species = int(cfg.get("n_species", 2))
        self.n_resources = int(cfg.get("n_resources", 2))

        if self.height < 1 or self.width_grid < 1:
            raise ValueError("CRM config requires height >= 1 and width_grid >= 1")
        if self.n_species < 1 or self.n_resources < 1:
            raise ValueError("CRM config requires n_species >= 1 and n_resources >= 1")

        self.dt = float(cfg.get("dt", 0.05))
        if self.dt <= 0.0:
            raise ValueError("CRM config requires dt > 0")

        self.dilution = float(cfg.get("dilution", 0.01))
        self.noise_std = float(cfg.get("noise_std", 0.0))

        reaction = cfg.get("reaction_matrix")
        if reaction is None:
            reaction = np.ones((self.n_species, self.n_resources), dtype=np.float32) * 0.25
        self.reaction_matrix = np.asarray(reaction, dtype=np.float32)
        if self.reaction_matrix.shape != (self.n_species, self.n_resources):
            raise ValueError("reaction_matrix must have shape (n_species, n_resources)")

        consumption = cfg.get("consumption_matrix")
        if consumption is None:
            consumption = np.ones((self.n_species, self.n_resources), dtype=np.float32) * 0.2
        self.consumption_matrix = np.asarray(consumption, dtype=np.float32)
        if self.consumption_matrix.shape != (self.n_species, self.n_resources):
            raise ValueError("consumption_matrix must have shape (n_species, n_resources)")

        resource_inflow = cfg.get("resource_inflow")
        if resource_inflow is None:
            resource_inflow = np.ones(self.n_resources, dtype=np.float32) * 0.2
        self.resource_inflow = np.asarray(resource_inflow, dtype=np.float32).reshape(-1)
        if self.resource_inflow.size != self.n_resources:
            raise ValueError("resource_inflow must have length n_resources")

        diff_species = cfg.get("diffusion_species", 0.02)
        if np.isscalar(diff_species):
            diff_species = np.full(self.n_species, float(diff_species), dtype=np.float32)
        self.diffusion_species = np.asarray(diff_species, dtype=np.float32).reshape(-1)
        if self.diffusion_species.size != self.n_species:
            raise ValueError("diffusion_species must be scalar or length n_species")

        diff_resources = cfg.get("diffusion_resources", 0.05)
        if np.isscalar(diff_resources):
            diff_resources = np.full(self.n_resources, float(diff_resources), dtype=np.float32)
        self.diffusion_resources = np.asarray(diff_resources, dtype=np.float32).reshape(-1)
        if self.diffusion_resources.size != self.n_resources:
            raise ValueError("diffusion_resources must be scalar or length n_resources")

        self.inject_scale = float(cfg.get("inject_scale", 0.05))

        # ── New: improved reservoir features ──────────────────────────
        self.inject_mode = str(cfg.get("inject_mode", "resource_only"))
        if self.inject_mode not in ("resource_only", "both"):
            raise ValueError("inject_mode must be 'resource_only' or 'both'")

        self.half_saturation = float(cfg.get("half_saturation", 0.0))
        if self.half_saturation < 0.0:
            raise ValueError("half_saturation must be >= 0")

        self.normalize_state = bool(cfg.get("normalize_state", False))
        self.cross_features = bool(cfg.get("cross_features", False))

        # Basal initialisation: when True the "zeros" reset mode seeds
        # resources at their inflow/dilution equilibrium and species at
        # a small non-zero level, so input perturbations immediately
        # drive nonlinear species–resource interactions instead of being
        # trapped in the X ≈ 0 dead zone.
        self.basal_init = bool(cfg.get("basal_init", False))
        self.basal_species = float(cfg.get("basal_species", 0.01))
        # ──────────────────────────────────────────────────────────────

        proj_cfg = dict(cfg.get("projection") or {})
        self._projection_cfg = CRMProjectionConfig(
            kind=str(proj_cfg.get("kind", "identity")).lower(),
            output_width=(
                int(proj_cfg["output_width"])
                if proj_cfg.get("output_width") is not None
                else None
            ),
            seed=int(proj_cfg.get("seed", 0)),
            scale=float(proj_cfg.get("scale", 1.0)),
        )

        self._cells = self.height * self.width_grid
        self._n_base_channels = self.n_species + self.n_resources
        self._n_cross_channels = (
            self.n_species * self.n_resources if self.cross_features else 0
        )
        self._raw_dim = (self._n_base_channels + self._n_cross_channels) * self._cells
        self._projection_matrix = self._build_projection_matrix()
        self.width = (
            self._raw_dim if self._projection_matrix is None else self._projection_matrix.shape[0]
        )

        self._X = np.zeros((self.n_species, self.height, self.width_grid), dtype=np.float32)
        self._R = np.zeros((self.n_resources, self.height, self.width_grid), dtype=np.float32)

    def _build_projection_matrix(self) -> np.ndarray | None:
        if self._projection_cfg.kind == "identity":
            return None
        if self._projection_cfg.kind != "random":
            raise ValueError("CRM projection.kind must be 'identity' or 'random'")
        out_w = self._projection_cfg.output_width
        if out_w is None or out_w < 1:
            raise ValueError("CRM random projection requires projection.output_width >= 1")
        rng = np.random.default_rng(self._projection_cfg.seed)
        std = self._projection_cfg.scale / np.sqrt(float(self._raw_dim))
        return rng.normal(0.0, std, size=(out_w, self._raw_dim)).astype(np.float32)

    def reset(self, rng: np.random.Generator, x0_mode: str = "zeros") -> None:
        if x0_mode == "zeros":
            if self.basal_init:
                # Resources at inflow/dilution equilibrium; species at a
                # small but non-zero level.  This lets the CRM respond to
                # injected perturbations through nonlinear growth rather
                # than sitting in the X ≈ 0 dead zone.
                r_eq = self.resource_inflow / max(self.dilution, 1e-12)
                self._R = np.broadcast_to(
                    r_eq[:, None, None],
                    (self.n_resources, self.height, self.width_grid),
                ).copy().astype(np.float32)
                self._X = np.full(
                    (self.n_species, self.height, self.width_grid),
                    self.basal_species,
                    dtype=np.float32,
                )
            else:
                self._X.fill(0.0)
                self._R.fill(0.0)
            return
        if x0_mode == "random":
            self._X = rng.random(self._X.shape, dtype=np.float32) * 0.1
            self._R = rng.random(self._R.shape, dtype=np.float32) * 0.1
            return
        raise ValueError("x0_mode must be 'zeros' or 'random'")

    def inject(
        self,
        input_values: np.ndarray,
        input_locations: np.ndarray,
        channel_idx: np.ndarray,
    ) -> None:
        scaled = input_values[channel_idx].astype(np.float32) * self.inject_scale
        flat = np.mod(input_locations, self._cells)
        rr = flat // self.width_grid
        cc = flat % self.width_grid

        if self.inject_mode == "both":
            # Vectorised dual injection into species AND resources.
            r_idx = channel_idx % self.n_resources
            np.add.at(self._R, (r_idx, rr, cc), scaled)
            s_idx = channel_idx % self.n_species
            np.add.at(self._X, (s_idx, rr, cc), scaled)
        else:
            if self.n_resources > 0:
                r_idx = channel_idx % self.n_resources
                np.add.at(self._R, (r_idx, rr, cc), scaled)
            elif self.n_species > 0:
                s_idx = channel_idx % self.n_species
                np.add.at(self._X, (s_idx, rr, cc), scaled)

    def step(self, rng: np.random.Generator) -> None:
        # ── Growth with optional Monod saturation ─────────────────────
        if self.half_saturation > 0.0:
            saturated_R = self._R / (self.half_saturation + self._R)
            growth_drive = np.einsum("sm,mhw->shw", self.reaction_matrix, saturated_R)
        else:
            growth_drive = np.einsum("sm,mhw->shw", self.reaction_matrix, self._R)

        dX = self.dt * (growth_drive * self._X - self.dilution * self._X)

        consumption = (
            np.einsum("sm,shw->mhw", self.consumption_matrix, self._X) * self._R
        )
        inflow = self.resource_inflow[:, None, None]
        dR = self.dt * (inflow - consumption - self.dilution * self._R)

        X_next = self._X + dX
        R_next = self._R + dR

        # Vectorised diffusion – no Python loop over channels.
        # _laplacian_periodic works on the full (channels, H, W) tensor
        # because np.roll and arithmetic broadcast over leading axes.
        X_next += (
            self.dt
            * self.diffusion_species[:, None, None]
            * _laplacian_periodic(self._X)
        )
        R_next += (
            self.dt
            * self.diffusion_resources[:, None, None]
            * _laplacian_periodic(self._R)
        )

        if self.noise_std > 0.0:
            X_next += rng.normal(0.0, self.noise_std, size=X_next.shape).astype(np.float32)
            R_next += rng.normal(0.0, self.noise_std, size=R_next.shape).astype(np.float32)

        self._X = np.clip(X_next, 0.0, None)
        self._R = np.clip(R_next, 0.0, None)

    def _build_raw_state(self) -> np.ndarray:
        """Assemble the full state vector from species, resources, and
        optional cross-features, with optional per-channel normalisation."""
        X_flat = self._X.reshape(self.n_species, -1)    # (S, cells)
        R_flat = self._R.reshape(self.n_resources, -1)   # (M, cells)

        # Vectorised assembly – no Python loops over channels
        parts = [X_flat, R_flat]
        if self.cross_features:
            # Outer product: (S, 1, cells) * (1, M, cells) → (S, M, cells)
            cross = (X_flat[:, None, :] * R_flat[None, :, :]).reshape(-1, self._cells)
            parts.append(cross)

        all_channels = np.concatenate(parts, axis=0)  # (total_channels, cells)

        if self.normalize_state:
            mu = all_channels.mean(axis=1, keepdims=True)
            std = all_channels.std(axis=1, keepdims=True)
            std = np.where(std > 1e-8, std, 1.0)
            all_channels = (all_channels - mu) / std

        return all_channels.reshape(-1).astype(np.float32)

    def get_state(self) -> np.ndarray:
        raw = self._build_raw_state()
        if self._projection_matrix is None:
            return raw
        return (self._projection_matrix @ raw).astype(np.float32)
