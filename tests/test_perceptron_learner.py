from __future__ import annotations

import numpy as np

from computingMicrobiome.evolution.config import LearnerConfig
from computingMicrobiome.evolution.learners import PerceptronGenotype, PerceptronLearner


def make_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    # Linearly separable by sign of first coordinate.
    y = (X[:, 0] > 0).astype(np.int8)
    return X, y


def test_perceptron_improves_on_linearly_separable_data() -> None:
    X, y = make_toy_dataset()
    cfg = LearnerConfig(
        learning_rate=0.1,
        l2=0.0,
        epochs=5,
        init_scale=0.1,
        normalize_features=True,
    )
    learner = PerceptronLearner(cfg)
    geno = PerceptronGenotype(
        learning_rate=cfg.learning_rate,
        l2=cfg.l2,
        epochs=cfg.epochs,
        init_scale=cfg.init_scale,
    )
    rng = np.random.default_rng(123)
    learner.reset_from_genotype(geno, rng)
    acc_before = learner.score(X, y)
    learner.fit(X, y)
    acc_after = learner.score(X, y)
    assert acc_after >= acc_before
    assert acc_after > 0.8


def test_reset_from_genotype_is_deterministic() -> None:
    X, y = make_toy_dataset()
    cfg = LearnerConfig(
        learning_rate=0.1,
        l2=0.0,
        epochs=1,
        init_scale=0.1,
        normalize_features=False,
    )
    geno = PerceptronGenotype(
        learning_rate=cfg.learning_rate,
        l2=cfg.l2,
        epochs=cfg.epochs,
        init_scale=cfg.init_scale,
    )

    rng1 = np.random.default_rng(1)
    learner1 = PerceptronLearner(cfg)
    learner1.reset_from_genotype(geno, rng1)
    learner1.fit(X, y)
    w1, b1 = learner1.weights.copy(), learner1.bias

    rng2 = np.random.default_rng(1)
    learner2 = PerceptronLearner(cfg)
    learner2.reset_from_genotype(geno, rng2)
    learner2.fit(X, y)
    w2, b2 = learner2.weights.copy(), learner2.bias

    assert np.allclose(w1, w2)
    assert b1 == b2

