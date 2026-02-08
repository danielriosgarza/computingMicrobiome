"""Linear SVM readout helper."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.svm import SVC


def make_linear_svm(config: Mapping[str, Any] | None = None) -> SVC:
    """Create a linear SVM readout with optional config overrides."""
    cfg = dict(config) if config is not None else {}
    cfg.pop("kernel", None)
    return SVC(kernel="linear", **cfg)
