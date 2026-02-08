import numpy as np

from computingMicrobiome.readouts.factory import make_readout
from computingMicrobiome.readouts.meta_evolutionary_linear import (
    FrozenLinearReadout,
    MetaEvolutionaryLinearReadout,
)


def _make_linearly_separable_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(320, 6))
    w = np.array([1.3, -2.1, 0.8, 0.0, 0.4, -0.6])
    logits = X @ w
    y = (logits > 0.0).astype(np.int8)
    return X, y


def test_meta_evo_readout_learns_separator():
    X, y = _make_linearly_separable_dataset()
    reg = MetaEvolutionaryLinearReadout(
        population_size=24,
        generations=40,
        tournament_size=4,
        elite_count=3,
        mutation_scale=0.25,
        support_fraction=0.65,
        min_adaptation_steps=3,
        max_adaptation_steps=18,
        adaptation_penalty=0.005,
        normalize_features=True,
        rng=np.random.default_rng(1),
    )
    reg.fit(X, y)
    assert reg.score(X, y) >= 0.9
    assert reg.best_genome_ is not None


def test_meta_evo_freeze_roundtrip_predictions():
    X, y = _make_linearly_separable_dataset(seed=2)
    reg = MetaEvolutionaryLinearReadout(
        population_size=18,
        generations=25,
        tournament_size=4,
        elite_count=2,
        min_adaptation_steps=2,
        max_adaptation_steps=12,
        normalize_features=True,
        rng=np.random.default_rng(3),
    )
    reg.fit(X, y)

    frozen = reg.freeze()
    payload = frozen.to_dict()
    loaded = FrozenLinearReadout.from_dict(payload)

    np.testing.assert_array_equal(frozen.predict(X), loaded.predict(X))


def test_frozen_save_load_json(tmp_path):
    X, y = _make_linearly_separable_dataset(seed=4)
    reg = MetaEvolutionaryLinearReadout(
        population_size=14,
        generations=16,
        tournament_size=4,
        elite_count=2,
        min_adaptation_steps=2,
        max_adaptation_steps=10,
        normalize_features=True,
        rng=np.random.default_rng(5),
    )
    reg.fit(X, y)
    frozen = reg.freeze()

    artifact = tmp_path / "frozen_readout.json"
    frozen.save_json(artifact)
    loaded = FrozenLinearReadout.load_json(artifact)

    np.testing.assert_array_equal(frozen.predict(X), loaded.predict(X))


def test_factory_meta_evo_alias():
    reg = make_readout(
        "meta_evo",
        config={
            "population_size": 8,
            "generations": 4,
            "min_adaptation_steps": 1,
            "max_adaptation_steps": 4,
        },
        rng=np.random.default_rng(0),
    )
    assert isinstance(reg, MetaEvolutionaryLinearReadout)
