"""IBM reservoir backend with pulse injection (toxin / popular metabolite) and left source.

Mirrors the notebook Task 4 setup: CROSS_FEED_6_SPECIES, fixed left-edge migrating
species, bit pulses at a single square (bit 0 = toxin, bit 1 = popular metabolite).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..ibm.diffusion import diffuse_resources
from ..ibm.dilution import apply_dilution
from ..ibm.encoding import encode_state
from ..ibm.metabolism import apply_maintenance, apply_uptake
from ..ibm.params import EnvParams, SpeciesParams, load_params
from ..ibm.pulse_injection import inject_bit_into_state
from ..ibm.reproduction import apply_reproduction
from ..ibm.state import GridState, make_zero_state
from ..ibm.stepper import tick
from ..ibm.universe import make_center_column_state
from .base import ReservoirBackend


class IBMReservoirBackendPulse(ReservoirBackend):
    """IBM reservoir with pulse injection (toxin/popular) and left source column."""

    def __init__(self, *, config: Mapping[str, Any] | None):
        cfg = dict(config or {})
        env, species = load_params(config)
        self.env: EnvParams = env
        self.species: SpeciesParams = species

        self.height = env.height
        self.width_grid = env.width_grid
        self.n_species = env.n_species
        self.n_resources = env.n_resources
        self._cells = self.height * self.width_grid

        # Pulse parameters (notebook defaults)
        self._center_r = int(cfg.get("pulse_center_r", self.height // 2))
        self._center_c = int(cfg.get("pulse_center_c", self.width_grid // 2))
        self._radius = int(cfg.get("pulse_radius", 2))
        self._toxin_conc = int(cfg.get("pulse_toxin_conc", 180))
        self._popular_conc = int(cfg.get("pulse_popular_conc", 200))

        # Left source (notebook: one species per row cycling)
        self._left_source_species = np.arange(self.height, dtype=np.int16) % self.n_species
        raw = cfg.get("left_source_species")
        if raw is not None:
            arr = np.asarray(raw, dtype=np.int16).reshape(-1)
            if arr.size != self.height:
                raise ValueError("left_source_species must have length height")
            if np.any((arr < -1) | (arr >= self.n_species)):
                raise ValueError("left_source_species must be -1 or in [0, n_species)")
            self._left_source_species = arr.copy()

        base = (
            species.yield_energy.astype(np.int32)
            + species.birth_energy.astype(np.int32)
            + species.uptake_rate.astype(np.int32)
            - species.maint_cost.astype(np.int32)
        )
        self._left_source_competition = np.maximum(base, 0).astype(np.int32)
        self._left_source_settle_energy = np.clip(
            species.birth_energy.astype(np.int32), 0, int(env.Emax)
        )
        self._left_source_margin = int(cfg.get("left_source_outcompete_margin", 1))
        self._left_source_colonize_empty = bool(cfg.get("left_source_colonize_empty", True))

        # State width = cells only (no trace)
        self.width = self._cells

        self._state = make_zero_state(
            height=self.height,
            width_grid=self.width_grid,
            n_resources=self.n_resources,
        )
        self._inject_count = 0

    def _enforce_left_source_column(self) -> None:
        self._state.occ[:, 0] = self._left_source_species
        self._state.E[:, 0] = 0
        self._state.R[:, :, 0] = 0

    def _exclude_left_source_from_dynamics(self) -> None:
        self._state.occ[:, 0] = -1
        self._state.E[:, 0] = 0
        self._state.R[:, :, 0] = 0

    def _apply_left_source_migration(self) -> None:
        if self.width_grid < 2:
            return
        tgt_col = 1
        occ = self._state.occ
        E_work = self._state.E.astype(np.int32, copy=True)
        source = self._left_source_species

        if self._left_source_colonize_empty:
            empty = occ[:, tgt_col] < 0
            can_seed = (source >= 0) & empty
            if np.any(can_seed):
                rows = np.where(can_seed)[0]
                src_species = source[rows]
                occ[rows, tgt_col] = src_species
                E_work[rows, tgt_col] = self._left_source_settle_energy[src_species]

        tgt = occ[:, tgt_col]
        can_compete = (source >= 0) & (tgt >= 0) & (source != tgt)
        if np.any(can_compete):
            rows = np.where(can_compete)[0]
            src_species = source[rows]
            tgt_species = tgt[rows]
            src_score = self._left_source_competition[src_species]
            tgt_score = self._left_source_competition[tgt_species]
            wins = src_score >= (tgt_score + self._left_source_margin)
            if np.any(wins):
                rows_win = rows[wins]
                src_win = source[rows_win]
                occ[rows_win, tgt_col] = src_win
                E_work[rows_win, tgt_col] = self._left_source_settle_energy[src_win]

        E_work[occ < 0] = 0
        cap = np.where(occ >= 0, self.species.energy_capacity[occ], 0)
        self._state.E = np.minimum(np.maximum(E_work, 0), cap).astype(np.uint8)
        self._enforce_left_source_column()

    def reset(self, rng: np.random.Generator, x0_mode: str = "zeros") -> None:
        if x0_mode != "zeros":
            raise ValueError("IBMReservoirBackendPulse only supports x0_mode='zeros'")
        state = make_center_column_state(
            self.env, species_id=0, energy_mean=5.0, rng=rng
        )
        center = self.width_grid // 2
        max_band = min(self.n_species, 5)
        for r in range(self.height):
            state.occ[r, center] = r % max_band
        self._state = state
        self._enforce_left_source_column()
        self._inject_count = 0

    def inject(
        self,
        input_values: np.ndarray,
        input_locations: np.ndarray,
        channel_idx: np.ndarray,
    ) -> None:
        # Only apply pulse for the first 8 ticks (8-bit memory task input phase).
        if self._inject_count >= 8:
            return
        bit = 0
        if input_values.size > 0:
            bit = int(np.clip(input_values.reshape(-1)[0], 0, 1))
        inject_bit_into_state(
            self._state,
            self.env,
            self.species,
            bit=bit,
            center_r=self._center_r,
            center_c=self._center_c,
            radius=self._radius,
            toxin_conc=self._toxin_conc,
            popular_conc=self._popular_conc,
        )
        self._inject_count += 1

    def step(self, rng: np.random.Generator) -> None:
        self._exclude_left_source_from_dynamics()
        diffuse_resources(self._state, self.env)
        apply_dilution(self._state, self.env, rng)
        apply_maintenance(self._state, self.species, self.env)
        apply_uptake(self._state, self.species, self.env)
        self._state.occ[:, 0] = -2
        apply_reproduction(self._state, self.species, self.env, rng)
        self._state.occ[:, 0] = -1
        self._state.E[self._state.occ < 0] = 0
        self._apply_left_source_migration()

    def get_state(self) -> np.ndarray:
        return encode_state(
            self._state, self.env, output_width=self.width
        ).astype(np.float32)
