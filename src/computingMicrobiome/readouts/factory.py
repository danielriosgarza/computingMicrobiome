"""Readout factory helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .evolutionary_linear import EvolutionaryLinearReadout
from .meta_evolutionary_linear import MetaEvolutionaryLinearReadout
from .moran_linear import MoranLinearReadout
from .svm_linear import make_linear_svm
from .base import Readout


def make_readout(
    kind: str = "svm",
    config: Mapping[str, Any] | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> Readout:
    """Create a readout by kind (svm, evo, meta_evo, moran)."""
    kind = kind.lower().strip()
    if kind == "svm":
        return make_linear_svm(config)
    if kind in {"evo", "evolution", "evolutionary"}:
        cfg = dict(config) if config is not None else {}
        if rng is not None and "rng" not in cfg:
            cfg["rng"] = rng
        return EvolutionaryLinearReadout(**cfg)
    if kind in {"meta_evo", "metaevo", "baldwin", "meta_evolutionary"}:
        cfg = dict(config) if config is not None else {}
        if rng is not None and "rng" not in cfg:
            cfg["rng"] = rng
        return MetaEvolutionaryLinearReadout(**cfg)
    if kind in {"moran", "steady_state", "steady-state", "moran_evo"}:
        cfg = dict(config) if config is not None else {}
        if rng is not None and "rng" not in cfg:
            cfg["rng"] = rng
        return MoranLinearReadout(**cfg)
    raise ValueError(f"Unknown readout kind: {kind}")
