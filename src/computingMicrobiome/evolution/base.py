"""Core protocols and shared types for evolution-of-learners framework.

This module intentionally stays dependency-light and only defines
interfaces / dataclasses that are reused across engines, learners,
and adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Tuple

import numpy as np


@runtime_checkable
class TaskSamplerProtocol(Protocol):
    """Protocol for task samplers used in the evolution framework.

    A task sampler is responsible for generating raw task instances for the
    inner (support) and outer (challenge) loops. Implementations are free to
    choose the internal representation of ``raw_x`` as long as they are
    compatible with the chosen ``RepresentationProtocol``.
    """

    def sample_support(
        self, rng: np.random.Generator, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a support set for inner-loop learning.

        Args:
            rng: NumPy random number generator.
            n_samples: Number of raw task instances to sample.

        Returns:
            A tuple ``(raw_x, raw_y)``. ``raw_x`` and ``raw_y`` have
            task-specific dtypes and shapes; the representation adapter is
            responsible for converting them into learner-ready features.
        """

    def sample_challenge(
        self, rng: np.random.Generator, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a challenge set for evaluation after inner-loop learning.

        The format mirrors :meth:`sample_support`.
        """


@runtime_checkable
class RepresentationProtocol(Protocol):
    """Protocol for representation adapters.

    A representation takes raw task instances produced by a task sampler
    and converts them into a feature matrix and target labels suitable
    for consumption by a :class:`LearnerProtocol` implementation.
    """

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - usually a no-op
        """Optional fitting stage for representations.

        Many representations (including the planned memory adapters) are
        stateless and can implement this as a no-op.
        """

    def transform(
        self, raw_x: np.ndarray, rng: np.random.Generator | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform raw task instances into features and labels.

        Args:
            raw_x: Raw task instances (task-specific structure).
            rng: Optional RNG for stochastic transformations.

        Returns:
            ``(X, y)`` where ``X`` is a 2D array of shape
            ``(n_rows, n_features)`` and ``y`` is a 1D array of binary
            labels compatible with the learner.
        """


@runtime_checkable
class LearnerProtocol(Protocol):
    """Protocol for inner-loop learners used in evolution experiments."""

    def reset_from_genotype(self, genotype, rng: np.random.Generator) -> None:
        """Reset internal parameters based on a genotype and RNG."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the learner on the provided feature matrix and labels."""

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return real-valued scores for each row in ``X``."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return discrete predictions for each row in ``X``."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return a scalar fitness/accuracy score for the given data."""


@dataclass
class IndividualState:
    """State of a single individual in the evolving population."""

    id: int
    genotype: object
    fitness: float | None = None
    pre_train_fitness: float | None = None
    post_train_fitness: float | None = None


@dataclass
class GenerationMetrics:
    """Summary metrics for a single generation of evolution."""

    generation: int
    mean_fitness: float
    best_fitness: float
    std_fitness: float
    mean_adaptation_gain: float
    replacement_rate: float


__all__ = [
    "TaskSamplerProtocol",
    "RepresentationProtocol",
    "LearnerProtocol",
    "IndividualState",
    "GenerationMetrics",
]

