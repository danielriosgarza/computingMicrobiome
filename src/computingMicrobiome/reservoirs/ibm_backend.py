"""IBM-backed reservoir implementation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..ibm.encoding import encode_state
from ..ibm.params import EnvParams, SpeciesParams, load_params
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

        self._inject_mode = str(cfg.get("inject_mode", "replace")).strip().lower()
        if self._inject_mode not in ("add", "replace"):
            raise ValueError("inject_mode must be 'add' or 'replace'")

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

        loc = np.mod(input_locations.astype(np.int64, copy=False), self._cells)
        rr = (loc // self.width_grid).astype(np.int64, copy=False)
        cc = (loc % self.width_grid).astype(np.int64, copy=False)

        ch = channel_idx.astype(np.int64, copy=False)
        vals = input_values.reshape(-1)
        scaled = np.rint(vals[ch % vals.size].astype(np.float32) * self.env.inject_scale)
        add = np.maximum(scaled, 0.0).astype(np.int32)

        if self.env.channel_to_resource is None:
            m_idx = np.mod(ch, self.n_resources).astype(np.int64)
        else:
            mapping = self.env.channel_to_resource
            m_idx = mapping[ch % mapping.size].astype(np.int64, copy=False)

        R_work = self._state.R.astype(np.int32, copy=True)
        if self._inject_mode == "replace":
            # Set resource at injection sites to the injected value so the
            # input signal is clearly visible (no adding on top of basal).
            R_work[m_idx, rr, cc] = add
        else:
            np.add.at(R_work, (m_idx, rr, cc), add)
        self._state.R = np.clip(R_work, 0, self.env.Rmax).astype(np.uint8)

    def step(self, rng: np.random.Generator) -> None:
        if self._trace_depth > 0:
            if self._trace_depth > 1:
                self._trace[1:] = self._trace[:-1] * self._trace_decay
            self._trace[0].fill(0.0)
        tick(self._state, self.env, self.species, rng)

    def get_state(self) -> np.ndarray:
        if self._state_width_mode == "raw":
            base = encode_state(self._state, self.env, output_width=self._raw_base_width)
        else:
            base = encode_state(self._state, self.env, output_width=self.width)
        if self._trace_depth <= 0:
            return base
        trace_flat = self._trace.reshape(-1).astype(np.float32, copy=False)
        return np.concatenate([base, trace_flat]).astype(np.float32)
