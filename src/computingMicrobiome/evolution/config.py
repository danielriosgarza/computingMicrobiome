"""Configuration dataclasses for the evolution-of-learners framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LearnerConfig:
    """Hyperparameters controlling inner-loop learning behaviour."""

    learning_rate: float
    l2: float
    epochs: int
    init_scale: float
    normalize_features: bool

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.l2 < 0.0:
            raise ValueError("l2 must be >= 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.init_scale <= 0.0:
            raise ValueError("init_scale must be > 0")


@dataclass
class EvolutionConfig:
    """Global configuration for the evolutionary outer loop."""

    population_size: int
    generations: int
    support_size: int
    challenge_size: int
    birth_events_per_generation: int
    parent_tournament_size: int
    death_tournament_size: int
    mutation_scale: float
    mutation_prob_per_gene: float
    accept_only_improving: bool
    seed: int

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be > 0")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.support_size <= 0:
            raise ValueError("support_size must be > 0")
        if self.challenge_size <= 0:
            raise ValueError("challenge_size must be > 0")
        if self.birth_events_per_generation <= 0:
            raise ValueError("birth_events_per_generation must be > 0")
        if self.parent_tournament_size < 2:
            raise ValueError("parent_tournament_size must be >= 2")
        if self.death_tournament_size < 2:
            raise ValueError("death_tournament_size must be >= 2")
        if not (0.0 <= self.mutation_prob_per_gene <= 1.0):
            raise ValueError("mutation_prob_per_gene must be in [0, 1]")
        if self.mutation_scale <= 0.0:
            raise ValueError("mutation_scale must be > 0")


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration used to reconstruct runs."""

    task_name: str
    representation_name: str
    learner_name: str
    engine_name: str
    task_params: Dict[str, Any] = field(default_factory=dict)
    representation_params: Dict[str, Any] = field(default_factory=dict)
    learner_params: Dict[str, Any] = field(default_factory=dict)
    evolution_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_name:
            raise ValueError("task_name must be non-empty")
        if not self.representation_name:
            raise ValueError("representation_name must be non-empty")
        if not self.learner_name:
            raise ValueError("learner_name must be non-empty")
        if not self.engine_name:
            raise ValueError("engine_name must be non-empty")


__all__ = [
    "LearnerConfig",
    "EvolutionConfig",
    "ExperimentConfig",
]

