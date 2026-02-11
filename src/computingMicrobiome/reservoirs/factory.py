"""Reservoir backend factory."""

from __future__ import annotations

from typing import Any, Mapping

from .base import ReservoirBackend
from .crm_backend import CRMReservoirBackend
from .eca_backend import ECAReservoirBackend


def make_reservoir(
    *,
    reservoir_kind: str = "eca",
    rule_number: int | None = None,
    width: int | None = None,
    boundary: str = "periodic",
    reservoir_config: Mapping[str, Any] | None = None,
) -> ReservoirBackend:
    """Create a reservoir backend by kind."""
    kind = str(reservoir_kind).strip().lower()
    if kind == "eca":
        if rule_number is None:
            raise ValueError("ECA reservoir requires rule_number")
        if width is None:
            raise ValueError("ECA reservoir requires width")
        return ECAReservoirBackend(rule_number=rule_number, width=width, boundary=boundary)
    if kind == "crm":
        return CRMReservoirBackend(config=reservoir_config)
    raise ValueError("reservoir_kind must be one of {'eca', 'crm'}")
