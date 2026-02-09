import numpy as np

from computingMicrobiome.readouts.factory import make_readout
from computingMicrobiome.readouts.moran_linear import MoranLinearReadout


def test_moran_readout_learns_linear_separator():
    rng = np.random.default_rng(0)
    n = 240
    X = rng.normal(size=(n, 3))
    w = np.array([1.8, -1.2, 0.6], dtype=float)
    y = ((X @ w) > 0.0).astype(np.int8)

    reg = MoranLinearReadout(
        population_size=64,
        generations=700,
        tournament_size=5,
        death_tournament_size=5,
        mutation_scale=0.08,
        mutation_schedule="linear_decay",
        min_mutation_scale=0.01,
        mutation_fraction=0.3,
        normalize_features=True,
        rng=rng,
    )
    reg.fit(X, y)
    acc = reg.score(X, y)
    assert acc > 0.98


def test_factory_moran_alias():
    reg = make_readout(
        "moran",
        config={
            "population_size": 10,
            "generations": 5,
            "mutation_fraction": 0.2,
        },
        rng=np.random.default_rng(0),
    )
    assert isinstance(reg, MoranLinearReadout)
