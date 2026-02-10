from __future__ import annotations

import numpy as np

from computingMicrobiome.benchmarks.k_bit_memory_bm import (
    build_dataset_output_window_only,
    run_episode_record,
    true_bits_from_episode_outputs,
)
from computingMicrobiome.evolution.adapters import (
    DirectMemoryRepresentation,
    MemoryTaskSampler,
    ReservoirMemoryRepresentation,
)


def test_direct_memory_representation_shapes() -> None:
    bits = 4
    sampler = MemoryTaskSampler(bits=bits, seed=0)
    rng = np.random.default_rng(0)
    raw_x, _ = sampler.sample_support(rng, n_samples=3)

    rep = DirectMemoryRepresentation(bits=bits)
    X, y = rep.transform(raw_x, rng)

    assert X.shape == (3 * bits, 2 * bits)
    assert y.shape == (3 * bits,)


def test_reservoir_memory_representation_shapes_and_labels() -> None:
    bits = 3
    rule_number = 110
    width = 100
    boundary = "periodic"
    recurrence = 2
    itr = 1
    d_period = 20

    sampler = MemoryTaskSampler(bits=bits, seed=1)
    rng = np.random.default_rng(1)
    raw_x, _ = sampler.sample_support(rng, n_samples=2)

    rep = ReservoirMemoryRepresentation(
        bits=bits,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        recurrence=recurrence,
        itr=itr,
        d_period=d_period,
    )

    X, y = rep.transform(raw_x, rng)
    # Each episode contributes `bits` rows in the output window.
    assert X.shape[0] == raw_x.shape[0] * bits
    assert y.shape[0] == raw_x.shape[0] * bits

    # Sanity: labels are binary.
    X_ref, y_ref, _ = build_dataset_output_window_only(
        bits=bits,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        recurrence=recurrence,
        itr=itr,
        d_period=d_period,
        seed=123,
    )
    assert set(np.unique(y_ref)) <= {0, 1}
    assert set(np.unique(y)) <= {0, 1}


def test_reservoir_representation_matches_episode_reference_semantics() -> None:
    bits = 3
    rule_number = 110
    width = 80
    boundary = "periodic"
    recurrence = 2
    itr = 1
    d_period = 10

    raw_x = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )

    rep = ReservoirMemoryRepresentation(
        bits=bits,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        recurrence=recurrence,
        itr=itr,
        d_period=d_period,
        seed_input_locations=7,
    )

    rng_rep = np.random.default_rng(123)
    X_rep, y_rep = rep.transform(raw_x, rng_rep)

    rng_ref = np.random.default_rng(123)
    X_ref_rows = []
    y_ref_rows = []
    for bits_arr in raw_x:
        ep = run_episode_record(
            bits_arr=bits_arr,
            rule_number=rule_number,
            width=width,
            boundary=boundary,
            itr=itr,
            d_period=d_period,
            rng=rng_ref,
            input_locations=rep._input_locations,
            reg=None,
            collect_states=False,
            x0_mode="zeros",
        )
        X_ref_rows.append(ep["X_tick"][-bits:])
        y_ref_rows.append(true_bits_from_episode_outputs(ep["outputs_tick"], bits))

    X_ref = np.vstack(X_ref_rows).astype(float)
    y_ref = np.concatenate(y_ref_rows).astype(np.int8)

    assert X_rep.shape == X_ref.shape
    assert y_rep.shape == y_ref.shape
    np.testing.assert_array_equal(y_rep, y_ref)
    np.testing.assert_array_equal(X_rep, X_ref)

