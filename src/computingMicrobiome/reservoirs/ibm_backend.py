"""IBM-backed reservoir implementation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..ibm.diffusion import diffuse_resources
from ..ibm.dilution import apply_dilution
from ..ibm.encoding import encode_state
from ..ibm.injection import PULSE_BIT, RESOURCE_ADD, RESOURCE_REPLACE, normalize_inject_mode
from ..ibm.metabolism import apply_maintenance, apply_uptake
from ..ibm.params import EnvParams, SpeciesParams, load_params
from ..ibm.pulse_injection import inject_bit_into_state
from ..ibm.reproduction import apply_reproduction
from ..ibm.state import GridState, make_zero_state
from ..ibm.stepper import tick


class IBMReservoirBackend:
    """Reservoir backend for a 1-individual-per-cell microbiome IBM."""

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
        self._raw_base_width = (
            (self.n_species * self._cells)
            + self._cells
            + (self.n_resources * self._cells)
            + self.n_species
        )
        self._state_width_mode = str(cfg.get("state_width_mode", "cells")).strip().lower()
        if self._state_width_mode not in {"cells", "raw"}:
            raise ValueError("IBM state_width_mode must be one of {'cells', 'raw'}")

        self._trace_depth = int(cfg.get("input_trace_depth", 0))
        self._trace_channels = int(cfg.get("input_trace_channels", 4))
        self._trace_decay = float(cfg.get("input_trace_decay", 1.0))
        if self._trace_depth < 0:
            raise ValueError("input_trace_depth must be >= 0")
        if self._trace_channels < 1:
            raise ValueError("input_trace_channels must be >= 1")
        if not (0.0 <= self._trace_decay <= 1.0):
            raise ValueError("input_trace_decay must be in [0, 1]")
        if self._trace_depth > 0 and self._state_width_mode != "raw":
            raise ValueError("input_trace_depth requires state_width_mode='raw'")
        self._trace_width = self._trace_depth * self._trace_channels

        self._inject_mode = normalize_inject_mode(
            cfg.get("inject_mode"),
            default=RESOURCE_REPLACE,
        )
        self._pulse_radius = int(cfg.get("pulse_radius", 2))
        self._pulse_toxin_conc = int(cfg.get("pulse_toxin_conc", 180))
        self._pulse_popular_conc = int(cfg.get("pulse_popular_conc", 200))
        if self._pulse_radius < 0:
            raise ValueError("pulse_radius must be >= 0")

        self._left_source_enabled = bool(cfg.get("left_source_enabled", False))
        self._left_source_colonize_empty = bool(
            cfg.get("left_source_colonize_empty", True)
        )
        self._left_source_margin = int(cfg.get("left_source_outcompete_margin", 0))
        if self._left_source_margin < 0:
            raise ValueError("left_source_outcompete_margin must be >= 0")
        if self._left_source_enabled and self.width_grid < 2:
            raise ValueError("left_source_enabled requires width_grid >= 2")

        raw_source = cfg.get("left_source_species")
        if raw_source is None:
            self._left_source_species_cfg = None
        else:
            source = np.asarray(raw_source, dtype=np.int16).reshape(-1)
            if source.size != self.height:
                raise ValueError("left_source_species must have length height")
            if np.any(source < -1) or np.any(source >= self.n_species):
                raise ValueError(
                    "left_source_species entries must be -1 or in [0, n_species)"
                )
            self._left_source_species_cfg = source.copy()

        raw_comp = cfg.get("left_source_competition")
        if raw_comp is None:
            # Default outcompetition score from species traits.
            base = (
                self.species.yield_energy.astype(np.int32)
                + self.species.birth_energy.astype(np.int32)
                + self.species.uptake_rate.astype(np.int32)
                - self.species.maint_cost.astype(np.int32)
            )
            comp = np.maximum(base, 0)
        elif np.isscalar(raw_comp):
            comp = np.full(self.n_species, int(raw_comp), dtype=np.int32)
        else:
            comp = np.asarray(raw_comp, dtype=np.int32).reshape(-1)
            if comp.size != self.n_species:
                raise ValueError("left_source_competition must be scalar or length n_species")
        if np.any(comp < 0):
            raise ValueError("left_source_competition must be >= 0")
        self._left_source_competition = comp.astype(np.int32, copy=True)

        raw_settle = cfg.get("left_source_settle_energy")
        if raw_settle is None:
            settle = self.species.birth_energy.astype(np.int32)
        elif np.isscalar(raw_settle):
            settle = np.full(self.n_species, int(raw_settle), dtype=np.int32)
        else:
            settle = np.asarray(raw_settle, dtype=np.int32).reshape(-1)
            if settle.size != self.n_species:
                raise ValueError(
                    "left_source_settle_energy must be scalar or length n_species"
                )
        if np.any(settle < 0):
            raise ValueError("left_source_settle_energy must be >= 0")
        self._left_source_settle_energy = np.clip(
            settle, 0, int(self.env.Emax)
        ).astype(np.int32, copy=True)
        self._left_source_species = np.full(self.height, -1, dtype=np.int16)

        if self._state_width_mode == "cells":
            self.width = self._cells
        else:
            self.width = self._raw_base_width + self._trace_width

        self._state: GridState = make_zero_state(
            height=self.height,
            width_grid=self.width_grid,
            n_resources=self.n_resources,
        )
        self._trace = np.zeros((self._trace_depth, self._trace_channels), dtype=np.float32)

    def _init_left_source_from_state(self) -> None:
        if not self._left_source_enabled:
            return
        if self._left_source_species_cfg is not None:
            source = self._left_source_species_cfg.copy()
        else:
            source = self._state.occ[:, 0].astype(np.int16, copy=True)
            bad = (source < -1) | (source >= self.n_species)
            source[bad] = -1
        self._left_source_species = source
        self._enforce_left_source_column()

    def _enforce_left_source_column(self) -> None:
        if not self._left_source_enabled:
            return
        self._state.occ[:, 0] = self._left_source_species
        self._state.E[:, 0] = 0
        self._state.R[:, :, 0] = 0

    def _exclude_left_source_from_dynamics(self) -> None:
        if not self._left_source_enabled:
            return
        self._state.occ[:, 0] = -1
        self._state.E[:, 0] = 0
        self._state.R[:, :, 0] = 0

    def _apply_left_source_migration(self) -> None:
        if not self._left_source_enabled or self.width_grid < 2:
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
            wins = src_score >= (tgt_score + int(self._left_source_margin))
            if np.any(wins):
                rows_win = rows[wins]
                src_win = source[rows_win]
                occ[rows_win, tgt_col] = src_win
                E_work[rows_win, tgt_col] = self._left_source_settle_energy[src_win]

        E_work[occ < 0] = 0
        cap = np.where(occ >= 0, self.species.energy_capacity[occ], 0)
        self._state.E = np.minimum(np.maximum(E_work, 0), cap).astype(np.uint8)

    def reset(self, rng: np.random.Generator, x0_mode: str = "zeros") -> None:
        if self._trace_depth > 0:
            self._trace.fill(0.0)

        if x0_mode == "zeros":
            if not self.env.basal_init:
                self._state = make_zero_state(
                    height=self.height,
                    width_grid=self.width_grid,
                    n_resources=self.n_resources,
                )
                self._init_left_source_from_state()
                return

            rr, cc = np.indices((self.height, self.width_grid))
            if self.env.basal_pattern == "stripes":
                sid = (rr % self.n_species).astype(np.int16)
            else:
                sid = ((rr + cc) % self.n_species).astype(np.int16)

            if self.env.basal_occupancy >= 1.0:
                occupied = np.ones((self.height, self.width_grid), dtype=bool)
            elif self.env.basal_occupancy <= 0.0:
                occupied = np.zeros((self.height, self.width_grid), dtype=bool)
            else:
                # Deterministic occupancy mask for reproducible "zeros" episodes.
                key = (rr * 73856093 + cc * 19349663) % 1000
                occupied = key < int(self.env.basal_occupancy * 1000.0)

            occ = np.full((self.height, self.width_grid), -1, dtype=np.int16)
            occ[occupied] = sid[occupied]

            E = np.zeros((self.height, self.width_grid), dtype=np.uint8)
            if self.env.basal_energy > 0:
                E[occupied] = np.uint8(self.env.basal_energy)

            # Initialize resources using per-resource basal levels if available,
            # otherwise fall back to the scalar basal_resource.
            br_vec = getattr(self.env, "basal_resource_vec", None)
            if br_vec is not None:
                br = np.asarray(br_vec, dtype=np.uint8).reshape(self.n_resources)
                R = np.broadcast_to(
                    br[:, None, None],
                    (self.n_resources, self.height, self.width_grid),
                ).copy()
            else:
                R = np.full(
                    (self.n_resources, self.height, self.width_grid),
                    np.uint8(self.env.basal_resource),
                    dtype=np.uint8,
                )
            self._state = GridState(occ=occ, E=E, R=R)
            self._init_left_source_from_state()
            return
        if x0_mode == "random":
            occ = np.full((self.height, self.width_grid), -1, dtype=np.int16)
            occupied = rng.random((self.height, self.width_grid)) < 0.5
            sid = rng.integers(
                0,
                self.n_species,
                size=(self.height, self.width_grid),
                dtype=np.int16,
            )
            occ[occupied] = sid[occupied]

            E = np.zeros((self.height, self.width_grid), dtype=np.uint8)
            if np.any(occupied):
                rand_e = rng.integers(
                    1,
                    self.env.Emax + 1,
                    size=int(np.count_nonzero(occupied)),
                    dtype=np.uint16,
                ).astype(np.uint8)
                E[occupied] = rand_e

            R = rng.integers(
                0,
                self.env.Rmax + 1,
                size=(self.n_resources, self.height, self.width_grid),
                dtype=np.uint16,
            ).astype(np.uint8)
            self._state = GridState(occ=occ, E=E, R=R)
            self._init_left_source_from_state()
            return
        raise ValueError("x0_mode must be 'zeros' or 'random'")

    def inject(
        self,
        input_values: np.ndarray,
        input_locations: np.ndarray,
        channel_idx: np.ndarray,
    ) -> None:
        if self._trace_depth > 0:
            row0 = np.zeros(self._trace_channels, dtype=np.float32)
            vals = input_values.reshape(-1).astype(np.float32, copy=False)
            n = min(vals.size, self._trace_channels)
            if n > 0:
                row0[:n] = vals[:n]
            self._trace[0] = row0

        if input_locations.size == 0 or input_values.size == 0:
            return

        loc = np.mod(input_locations.astype(np.int64, copy=False).reshape(-1), self._cells)
        rr = (loc // self.width_grid).astype(np.int64, copy=False)
        cc = (loc % self.width_grid).astype(np.int64, copy=False)

        ch = channel_idx.astype(np.int64, copy=False).reshape(-1)
        if ch.size != loc.size:
            return

        vals = input_values.reshape(-1)
        if self._inject_mode == PULSE_BIT:
            vals_f = vals.astype(np.float64, copy=False)
            for k in range(loc.size):
                channel = int(ch[k] % vals_f.size)
                raw = float(vals_f[channel])
                bit = 1 if raw > 0.5 else 0
                inject_bit_into_state(
                    self._state,
                    self.env,
                    self.species,
                    bit=bit,
                    center_r=int(rr[k]),
                    center_c=int(cc[k]),
                    radius=self._pulse_radius,
                    toxin_conc=self._pulse_toxin_conc,
                    popular_conc=self._pulse_popular_conc,
                )
            return

        scaled = np.rint(vals[ch % vals.size].astype(np.float32) * self.env.inject_scale)
        add = np.maximum(scaled, 0.0).astype(np.int32)

        if self.env.channel_to_resource is None:
            m_idx = np.mod(ch, self.n_resources).astype(np.int64)
        else:
            mapping = self.env.channel_to_resource
            m_idx = mapping[ch % mapping.size].astype(np.int64, copy=False)

        R_work = self._state.R.astype(np.int32, copy=True)
        if self._inject_mode == RESOURCE_REPLACE:
            # Set resource at injection sites to the injected value so the
            # input signal is clearly visible (no adding on top of basal).
            R_work[m_idx, rr, cc] = add
        elif self._inject_mode == RESOURCE_ADD:
            np.add.at(R_work, (m_idx, rr, cc), add)
        else:
            raise RuntimeError(f"Unsupported inject_mode: {self._inject_mode}")
        self._state.R = np.clip(R_work, 0, self.env.Rmax).astype(np.uint8)

    def step(self, rng: np.random.Generator) -> None:
        if self._trace_depth > 0:
            if self._trace_depth > 1:
                self._trace[1:] = self._trace[:-1] * self._trace_decay
            self._trace[0].fill(0.0)

        if not self._left_source_enabled:
            tick(self._state, self.env, self.species, rng)
            return

        self._exclude_left_source_from_dynamics()

        # Explicit phase-by-phase update so we can keep the source column
        # outside the dynamics and only allow directed migration into column 1.
        diffuse_resources(self._state, self.env)
        apply_dilution(self._state, self.env, rng)
        apply_maintenance(self._state, self.species, self.env)
        apply_uptake(self._state, self.species, self.env)

        # Block reproduction targets into the source column.
        self._state.occ[:, 0] = -2
        apply_reproduction(self._state, self.species, self.env, rng)
        self._state.occ[:, 0] = -1
        self._state.E[self._state.occ < 0] = 0

        self._apply_left_source_migration()
        self._enforce_left_source_column()

    def get_state(self) -> np.ndarray:
        if self._state_width_mode == "raw":
            base = encode_state(self._state, self.env, output_width=self._raw_base_width)
        else:
            base = encode_state(self._state, self.env, output_width=self.width)
        if self._trace_depth <= 0:
            return base
        trace_flat = self._trace.reshape(-1).astype(np.float32, copy=False)
        return np.concatenate([base, trace_flat]).astype(np.float32)
