"""Genotype mutation operators for evolution-of-learners."""

from __future__ import annotations

import numpy as np

from .config import EvolutionConfig
from .learners import PerceptronGenotype


def _mutate_scalar(
    value: float, rng: np.random.Generator, scale: float, low: float, high: float
) -> float:
    """Mutate a scalar value with log-normal noise and clipping."""
    noise = rng.normal(loc=0.0, scale=scale)
    mutated = value * float(np.exp(noise))
    return float(np.clip(mutated, low, high))


def mutate_perceptron_genotype(
    genotype: PerceptronGenotype, cfg: EvolutionConfig, rng: np.random.Generator
) -> PerceptronGenotype:
    """Return a mutated copy of a :class:`PerceptronGenotype`.

    Each field is perturbed with probability ``cfg.mutation_prob_per_gene``.
    Float fields use multiplicative log-normal noise; the integer ``epochs``
    field is mutated by a bounded integer step.
    """
    p = float(cfg.mutation_prob_per_gene)
    scale = float(cfg.mutation_scale)

    lr = genotype.learning_rate
    if rng.random() < p:
        lr = _mutate_scalar(lr, rng, scale, low=1e-5, high=1.0)

    l2 = genotype.l2
    if rng.random() < p:
        l2 = _mutate_scalar(l2, rng, scale, low=0.0, high=1.0)

    epochs = genotype.epochs
    if rng.random() < p:
        step = int(rng.integers(-5, 6))
        epochs = int(np.clip(epochs + step, 1, 200))

    init_scale = genotype.init_scale
    if rng.random() < p:
        init_scale = _mutate_scalar(init_scale, rng, scale, low=1e-4, high=2.0)

    return PerceptronGenotype(
        learning_rate=lr,
        l2=l2,
        epochs=epochs,
        init_scale=init_scale,
    )


__all__ = ["mutate_perceptron_genotype"]

