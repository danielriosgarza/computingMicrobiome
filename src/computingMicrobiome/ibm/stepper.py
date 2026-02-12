"""IBM one-tick orchestration."""

from __future__ import annotations

import numpy as np

from .diffusion import diffuse_resources
from .dilution import apply_dilution
from .metabolism import apply_maintenance, apply_uptake
from .params import EnvParams, SpeciesParams
from .reproduction import apply_reproduction
from .state import GridState


def tick(
    state: GridState,
    env: EnvParams,
    species: SpeciesParams,
    rng: np.random.Generator,
) -> None:
    """Advance one synchronous IBM update."""
    diffuse_resources(state, env)
    apply_dilution(state, env, rng)
    apply_maintenance(state, species, env)
    apply_uptake(state, species, env)
    apply_reproduction(state, species, env, rng)
