"""CRM-backed reservoir implementation (MVP)."""

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
        self._raw_dim = (self.n_species + self.n_resources) * self._cells
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
        vals = input_values[channel_idx].astype(np.float32)
        flat = np.mod(input_locations, self._cells)
        rr = flat // self.width_grid
        cc = flat % self.width_grid

        for k in range(vals.size):
            v = vals[k]
            if self.n_resources > 0:
                m = int(channel_idx[k] % self.n_resources)
                self._R[m, rr[k], cc[k]] += self.inject_scale * v
            if self.n_species > 0 and self.n_resources == 0:
                s = int(channel_idx[k] % self.n_species)
                self._X[s, rr[k], cc[k]] += self.inject_scale * v

    def step(self, rng: np.random.Generator) -> None:
        growth_drive = np.einsum("sm,mhw->shw", self.reaction_matrix, self._R)
        dX = self.dt * (growth_drive * self._X - self.dilution * self._X)

        consumption = (
            np.einsum("sm,shw->mhw", self.consumption_matrix, self._X) * self._R
        )
        inflow = self.resource_inflow[:, None, None]
        dR = self.dt * (inflow - consumption - self.dilution * self._R)

        X_next = self._X + dX
        R_next = self._R + dR

        for s in range(self.n_species):
            X_next[s] += self.dt * self.diffusion_species[s] * _laplacian_periodic(self._X[s])
        for m in range(self.n_resources):
            R_next[m] += self.dt * self.diffusion_resources[m] * _laplacian_periodic(self._R[m])

        if self.noise_std > 0.0:
            X_next += rng.normal(0.0, self.noise_std, size=X_next.shape).astype(np.float32)
            R_next += rng.normal(0.0, self.noise_std, size=R_next.shape).astype(np.float32)

        self._X = np.clip(X_next, 0.0, None)
        self._R = np.clip(R_next, 0.0, None)

    def get_state(self) -> np.ndarray:
        raw = np.concatenate([self._X.reshape(-1), self._R.reshape(-1)], axis=0).astype(
            np.float32
        )
        if self._projection_matrix is None:
            return raw
        return (self._projection_matrix @ raw).astype(np.float32)
