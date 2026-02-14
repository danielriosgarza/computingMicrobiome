"""Shared injection-mode handling for IBM reservoirs."""

from __future__ import annotations

from typing import Final


RESOURCE_ADD: Final[str] = "resource_add"
RESOURCE_REPLACE: Final[str] = "resource_replace"
PULSE_BIT: Final[str] = "pulse_bit"

_ALIAS_TO_MODE: Final[dict[str, str]] = {
    "resource_add": RESOURCE_ADD,
    "add": RESOURCE_ADD,
    "resource_replace": RESOURCE_REPLACE,
    "replace": RESOURCE_REPLACE,
    "pulse_bit": PULSE_BIT,
    "pulse": PULSE_BIT,
}


def normalize_inject_mode(raw_mode: object | None, *, default: str) -> str:
    """Return canonical IBM inject mode with backward-compatible aliases."""
    mode = str(default if raw_mode is None else raw_mode).strip().lower()
    canonical = _ALIAS_TO_MODE.get(mode)
    if canonical is None:
        raise ValueError(
            "inject_mode must be one of {'resource_add', 'resource_replace', "
            "'pulse_bit'} (aliases: {'add', 'replace', 'pulse'})"
        )
    return canonical

