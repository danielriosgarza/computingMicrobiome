import numpy as np

from computingMicrobiome.readouts.digital_linear import DigitalLinearReadout
from computingMicrobiome.readouts.factory import make_readout


def test_digital_readout_learns_linear_separator():
    rng = np.random.default_rng(0)
    n = 220
    X = rng.normal(size=(n, 3))
    w = np.array([1.6, -1.0, 0.5], dtype=float)
    y = ((X @ w) > 0.0).astype(np.int8)

    reg = DigitalLinearReadout(
        population_size=64,
        generations=2000,
        tournament_size=5,
        death_tournament_size=5,
        min_genes=2,
        max_genes=16,
        init_genes=8,
        mutation_rate_feature=0.35,
        mutation_rate_weight=0.35,
        mutation_rate_bias=0.2,
        mutation_rate_insert=0.1,
        mutation_rate_delete=0.1,
        mutation_rate_swap=0.05,
        complexity_penalty=0.0005,
        accept_only_improving=False,
        normalize_features=True,
        rng=np.random.default_rng(1),
    )
    reg.fit(X, y)
    acc = reg.score(X, y)
    assert acc > 0.95
    assert reg.best_genome_ is not None


def test_factory_digital_alias():
    reg = make_readout(
        "digital",
        config={
            "population_size": 12,
            "generations": 10,
            "min_genes": 1,
            "max_genes": 6,
        },
        rng=np.random.default_rng(0),
    )
    assert isinstance(reg, DigitalLinearReadout)
