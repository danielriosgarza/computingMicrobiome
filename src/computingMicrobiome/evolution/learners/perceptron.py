"""Perceptron-like linear learner for evolution-of-learners experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import LearnerProtocol
from ..config import LearnerConfig


@dataclass
class PerceptronGenotype:
    """Genotype controlling the behaviour of :class:`PerceptronLearner`."""

    learning_rate: float
    l2: float
    epochs: int
    init_scale: float


class PerceptronLearner(LearnerProtocol):
    """Simple linear classifier with perceptron-style updates."""

    def __init__(self, cfg: LearnerConfig) -> None:
        self._cfg = cfg
        self._w: np.ndarray | None = None
        self._b: float | None = None
        self._rng: np.random.Generator | None = None
        self._norm_mean: np.ndarray | None = None
        self._norm_std: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray | None:
        return self._w

    @property
    def bias(self) -> float | None:
        return self._b

    def reset_from_genotype(
        self, genotype: PerceptronGenotype, rng: np.random.Generator
    ) -> None:
        """Initialise weights and bias from a genotype."""
        self._rng = rng
        self._cfg = LearnerConfig(
            learning_rate=genotype.learning_rate,
            l2=genotype.l2,
            epochs=genotype.epochs,
            init_scale=genotype.init_scale,
            normalize_features=self._cfg.normalize_features,
        )
        # Weights initialised lazily on first fit when input dimension is known.
        self._w = None
        self._b = 0.0
        self._norm_mean = None
        self._norm_std = None

    def _ensure_params(self, n_features: int) -> None:
        if self._w is None:
            rng = self._rng or np.random.default_rng()
            scale = float(self._cfg.init_scale)
            self._w = rng.normal(loc=0.0, scale=scale, size=n_features)
            self._b = 0.0

    def _prepare_X_for_fit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._cfg.normalize_features:
            # Simple standardisation per-feature with numerical guards.
            self._norm_mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0.0] = 1.0
            self._norm_std = std
            X = (X - self._norm_mean) / self._norm_std
        return X

    def _prepare_X_for_inference(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if (
            self._cfg.normalize_features
            and self._norm_mean is not None
            and self._norm_std is not None
        ):
            X = (X - self._norm_mean) / self._norm_std
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train using a perceptron-style update rule.

        ``y`` is expected to be in ``{0, 1}``; internally it is mapped to
        ``{-1, +1}`` for the updates.
        """
        X = self._prepare_X_for_fit(X)
        y = np.asarray(y, dtype=int).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have compatible shapes")

        self._ensure_params(X.shape[1])
        assert self._w is not None
        assert self._b is not None

        rng = self._rng or np.random.default_rng()
        y_signed = np.where(y > 0, 1.0, -1.0)
        lr = float(self._cfg.learning_rate)
        l2 = float(self._cfg.l2)

        indices = np.arange(X.shape[0])
        for _ in range(int(self._cfg.epochs)):
            rng.shuffle(indices)
            for idx in indices:
                xi = X[idx]
                yi = y_signed[idx]
                margin = yi * (np.dot(self._w, xi) + self._b)
                if margin <= 0.0:
                    # Mistake: update weights and bias.
                    self._w += lr * yi * xi
                    self._b += lr * yi
                # L2 shrinkage (ridge-style) for stability.
                if l2 > 0.0:
                    self._w *= max(0.0, 1.0 - lr * l2)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_X_for_inference(X)
        if self._w is None:
            self._ensure_params(X.shape[1])
        if self._w is None or self._b is None:
            raise RuntimeError("Learner has not been initialised via reset_from_genotype")
        return X @ self._w + self._b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(np.int8)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=int).reshape(-1)
        if y.size == 0:
            return 0.0
        preds = self.predict(X)
        correct = (preds == y).sum()
        return float(correct) / float(y.size)


__all__ = ["PerceptronGenotype", "PerceptronLearner"]

