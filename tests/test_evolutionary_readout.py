import numpy as np

from computingMicrobiome.readouts.evolutionary_linear import EvolutionaryLinearReadout
from computingMicrobiome.feature_sources import build_reservoir_opcode_logic16_dataset
from experiments.compare_readouts import _infer_label_mode


def test_evo_readout_learns_linear_separator():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    w = np.array([1.5, -2.0])
    y = ((X @ w) > 0).astype(np.int8)

    reg = EvolutionaryLinearReadout(
        population_size=64,
        generations=200,
        tournament_size=5,
        elite_count=4,
        mutation_scale=0.1,
        batch_size=None,
        normalize_features=True,
        rng=rng,
    )
    reg.fit(X, y)
    acc = reg.score(X, y)
    assert acc > 0.99


def test_evo_readout_opcode_logic16_reservoir():
    X, y, _ = build_reservoir_opcode_logic16_dataset(
        rule_number=110,
        width=128,
        boundary="periodic",
        recurrence=4,
        itr=4,
        d_period=10,
        repeats=1,
        feature_mode="cue_tick",
        output_window=2,
        seed=0,
    )
    reg = EvolutionaryLinearReadout(
        population_size=128,
        generations=300,
        tournament_size=5,
        elite_count=6,
        mutation_scale=0.08,
        batch_size=None,
        normalize_features=True,
        rng=np.random.default_rng(0),
    )
    reg.fit(X, y)
    acc = reg.score(X, y)
    assert acc >= 0.95


def test_label_mode_inference():
    y_bin = np.array([0, 1, 1, 0], dtype=np.int8)
    info_bin = _infer_label_mode(y_bin)
    assert info_bin["y_mode"] == "binary"

    y_multi = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int8)
    info_multi = _infer_label_mode(y_multi)
    assert info_multi["y_mode"] == "multilabel"
