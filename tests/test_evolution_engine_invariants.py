from __future__ import annotations

import numpy as np

from computingMicrobiome.evolution.adapters import (
    DirectMemoryRepresentation,
    MemoryTaskSampler,
)
from computingMicrobiome.evolution.config import EvolutionConfig, LearnerConfig
from computingMicrobiome.evolution.engines import MoranEvolutionEngine
from computingMicrobiome.evolution.learners import PerceptronGenotype, PerceptronLearner


def _make_engine_and_components(seed: int = 0):
    bits = 2
    task_sampler = MemoryTaskSampler(bits=bits, seed=seed)
    representation = DirectMemoryRepresentation(bits=bits)
    learner_cfg = LearnerConfig(
        learning_rate=0.1,
        l2=0.0,
        epochs=1,
        init_scale=0.1,
        normalize_features=True,
    )

    def learner_factory() -> PerceptronLearner:
        return PerceptronLearner(learner_cfg)

    evo_cfg = EvolutionConfig(
        population_size=5,
        generations=3,
        support_size=4,
        challenge_size=4,
        birth_events_per_generation=5,
        parent_tournament_size=2,
        death_tournament_size=2,
        mutation_scale=0.1,
        mutation_prob_per_gene=0.5,
        accept_only_improving=False,
        seed=seed,
    )
    base_genotype = PerceptronGenotype(
        learning_rate=learner_cfg.learning_rate,
        l2=learner_cfg.l2,
        epochs=learner_cfg.epochs,
        init_scale=learner_cfg.init_scale,
    )
    initial_genotypes = [base_genotype for _ in range(evo_cfg.population_size)]
    engine = MoranEvolutionEngine()
    rng = np.random.default_rng(seed)
    return engine, task_sampler, representation, learner_factory, evo_cfg, rng, initial_genotypes


def test_population_size_invariant_and_no_nans_in_history() -> None:
    (
        engine,
        task_sampler,
        representation,
        learner_factory,
        evo_cfg,
        rng,
        initial_genotypes,
    ) = _make_engine_and_components(seed=0)

    res = engine.run(
        task_sampler=task_sampler,
        representation=representation,
        learner_factory=learner_factory,
        evolution_config=evo_cfg,
        rng=rng,
        initial_genotypes=initial_genotypes,
    )

    assert len(res.history) == evo_cfg.generations
    assert res.final_population_fitness.shape[0] == evo_cfg.population_size
    assert not np.isnan(res.final_population_fitness).any()


def test_determinism_with_fixed_seed() -> None:
    components1 = _make_engine_and_components(seed=42)
    res1 = components1[0].run(
        task_sampler=components1[1],
        representation=components1[2],
        learner_factory=components1[3],
        evolution_config=components1[4],
        rng=components1[5],
        initial_genotypes=components1[6],
    )

    components2 = _make_engine_and_components(seed=42)
    res2 = components2[0].run(
        task_sampler=components2[1],
        representation=components2[2],
        learner_factory=components2[3],
        evolution_config=components2[4],
        rng=components2[5],
        initial_genotypes=components2[6],
    )

    hist1 = [m.best_fitness for m in res1.history]
    hist2 = [m.best_fitness for m in res2.history]
    assert hist1 == hist2

