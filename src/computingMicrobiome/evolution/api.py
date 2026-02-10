"""Public API for running evolution-of-learners experiments."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .adapters import (
    DirectMemoryRepresentation,
    MemoryTaskSampler,
    ReservoirMemoryRepresentation,
)
from .config import EvolutionConfig, ExperimentConfig, LearnerConfig
from .engines import MoranEvolutionEngine
from .learners import PerceptronGenotype, PerceptronLearner
from .results import EvolutionRunResult


def _make_task_sampler(task: str, task_params: Dict[str, Any]) -> MemoryTaskSampler:
    if task != "memory":
        raise ValueError(f"Unsupported task: {task!r}")
    bits = int(task_params.get("bits", 8))
    seed = int(task_params.get("seed", 0))
    return MemoryTaskSampler(bits=bits, seed=seed)


def _make_representation(
    representation: str, bits: int, representation_params: Dict[str, Any]
):
    if representation == "direct":
        return DirectMemoryRepresentation(bits=bits)
    if representation == "reservoir":
        return ReservoirMemoryRepresentation(
            bits=bits,
            rule_number=int(representation_params.get("rule_number", 110)),
            width=int(representation_params.get("width", 700)),
            boundary=str(representation_params.get("boundary", "periodic")),
            recurrence=int(representation_params.get("recurrence", 4)),
            itr=int(representation_params.get("itr", 2)),
            d_period=int(representation_params.get("d_period", 200)),
        )
    raise ValueError(f"Unsupported representation: {representation!r}")


def _make_learner_factory(
    learner: str, learner_params: Dict[str, Any]
):
    if learner != "perceptron":
        raise ValueError(f"Unsupported learner: {learner!r}")
    cfg = LearnerConfig(
        learning_rate=float(learner_params.get("learning_rate", 0.05)),
        l2=float(learner_params.get("l2", 1e-4)),
        epochs=int(learner_params.get("epochs", 5)),
        init_scale=float(learner_params.get("init_scale", 0.1)),
        normalize_features=bool(learner_params.get("normalize_features", True)),
    )

    def factory() -> PerceptronLearner:
        return PerceptronLearner(cfg)

    return factory


def _make_evolution_config(
    n_generations: int,
    population_size: int,
    support_size: int,
    challenge_size: int,
    seed: int,
    evolution_params: Dict[str, Any],
) -> EvolutionConfig:
    return EvolutionConfig(
        population_size=population_size,
        generations=n_generations,
        support_size=support_size,
        challenge_size=challenge_size,
        birth_events_per_generation=int(
            evolution_params.get("birth_events_per_generation", population_size)
        ),
        parent_tournament_size=int(
            evolution_params.get("parent_tournament_size", 2)
        ),
        death_tournament_size=int(
            evolution_params.get("death_tournament_size", 2)
        ),
        mutation_scale=float(evolution_params.get("mutation_scale", 0.1)),
        mutation_prob_per_gene=float(
            evolution_params.get("mutation_prob_per_gene", 0.5)
        ),
        accept_only_improving=bool(
            evolution_params.get("accept_only_improving", False)
        ),
        seed=seed,
    )


def run_evolution_of_learners(
    task: str,
    representation: str,
    learner: str = "perceptron",
    engine: str = "moran",
    *,
    n_generations: int = 200,
    population_size: int = 64,
    support_size: int = 128,
    challenge_size: int = 128,
    seed: int = 0,
    task_params: Dict[str, Any] | None = None,
    representation_params: Dict[str, Any] | None = None,
    learner_params: Dict[str, Any] | None = None,
    evolution_params: Dict[str, Any] | None = None,
    progress: bool = False,
    progress_mode: str = "auto",
    progress_every: int = 10,
) -> EvolutionRunResult:
    """High-level entry point for running an evolution-of-learners experiment.

    Args:
        progress: If True, show run progress.
        progress_mode: One of {"auto", "tqdm", "print"}.
            - "auto": use tqdm if available, otherwise print.
            - "tqdm": require tqdm and show a progress bar.
            - "print": periodic text progress.
        progress_every: Generation interval for text progress when using print mode.
    """
    if engine != "moran":
        raise ValueError(f"Unsupported engine: {engine!r}")

    task_params = dict(task_params or {})
    representation_params = dict(representation_params or {})
    learner_params = dict(learner_params or {})
    evolution_params = dict(evolution_params or {})

    bits = int(task_params.get("bits", 8))

    rng = np.random.default_rng(seed)

    task_sampler = _make_task_sampler(task, task_params)
    representation_obj = _make_representation(representation, bits, representation_params)
    learner_factory = _make_learner_factory(learner, learner_params)
    evo_cfg = _make_evolution_config(
        n_generations=n_generations,
        population_size=population_size,
        support_size=support_size,
        challenge_size=challenge_size,
        seed=seed,
        evolution_params=evolution_params,
    )

    # Initial population: start from a common base genotype.
    base_genotype = PerceptronGenotype(
        learning_rate=float(learner_params.get("learning_rate", 0.05)),
        l2=float(learner_params.get("l2", 1e-4)),
        epochs=int(learner_params.get("epochs", 5)),
        init_scale=float(learner_params.get("init_scale", 0.1)),
    )
    initial_genotypes = [
        PerceptronGenotype(
            learning_rate=base_genotype.learning_rate,
            l2=base_genotype.l2,
            epochs=base_genotype.epochs,
            init_scale=base_genotype.init_scale,
        )
        for _ in range(population_size)
    ]

    engine_obj = MoranEvolutionEngine()
    result = engine_obj.run(
        task_sampler=task_sampler,
        representation=representation_obj,
        learner_factory=learner_factory,
        evolution_config=evo_cfg,
        rng=rng,
        initial_genotypes=initial_genotypes,
        progress=progress,
        progress_mode=progress_mode,
        progress_every=progress_every,
    )

    exp_cfg = ExperimentConfig(
        task_name=task,
        representation_name=representation,
        learner_name=learner,
        engine_name=engine,
        task_params=task_params,
        representation_params=representation_params,
        learner_params=learner_params,
        evolution_params=evolution_params,
    )

    result.config["experiment"] = exp_cfg.__dict__
    return result


__all__ = ["run_evolution_of_learners"]

