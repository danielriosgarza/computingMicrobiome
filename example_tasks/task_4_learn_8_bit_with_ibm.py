"""Task 4: Learn the 8-bit memory task with the IBM reservoir backend."""

from __future__ import annotations

import numpy as np

from computingMicrobiome.benchmarks.k_bit_memory_bm import (
    build_dataset_output_window_only,
    evaluate_memory_trials,
)
from computingMicrobiome.readouts.factory import make_readout

BITS = 8
BOUNDARY = "periodic"
RECURRENCE = 4
ITR = 12
D_PERIOD = 8
N_TRIALS = 100

# Delay in simulation steps from bit injection to output window.
# For this benchmark, each tick is separated by (ITR + 1) steps.
TRACE_DEPTH = (D_PERIOD + BITS) * (ITR + 1) + 8

IBM_CFG = {
    "height": 8,
    "width_grid": 8,
    "n_species": 1,
    "n_resources": 1,
    "Rmax": 255,
    "Emax": 255,
    "dilution_p": 0.0,
    "inject_scale": 0.0,
    "diff_numer": 0,
    "state_width_mode": "raw",
    "input_trace_depth": TRACE_DEPTH,
    "input_trace_channels": 4,
    "input_trace_decay": 1.0,
    "maint_cost": [0],
    "uptake_rate": [0],
    "yield_energy": [0],
    "div_threshold": [255],
    "div_cost": [0],
    "birth_energy": [0],
}


def main() -> None:
    width = int(IBM_CFG["height"]) * int(IBM_CFG["width_grid"])
    seed = 0

    X, y, input_locations = build_dataset_output_window_only(
        bits=BITS,
        rule_number=110,
        width=width,
        boundary=BOUNDARY,
        recurrence=RECURRENCE,
        itr=ITR,
        d_period=D_PERIOD,
        seed=seed,
        reservoir_kind="ibm",
        reservoir_config=IBM_CFG,
    )

    rng = np.random.default_rng(seed)
    reg = make_readout("svm", {"C": 10.0, "class_weight": "balanced"}, rng=rng)
    reg.fit(X, y)
    train_acc = float(reg.score(X, y))

    correctness = evaluate_memory_trials(
        reg=reg,
        bits=BITS,
        rule_number=110,
        width=width,
        boundary=BOUNDARY,
        recurrence=RECURRENCE,
        itr=ITR,
        d_period=D_PERIOD,
        input_locations=input_locations,
        n_trials=N_TRIALS,
        seed_trials=42,
        reservoir_kind="ibm",
        reservoir_config=IBM_CFG,
    )
    trial_acc = (correctness == 1).mean(axis=1)

    print(f"IBM training set: X={X.shape}, y={y.shape}")
    print(f"IBM train accuracy: {train_acc:.4f}")
    print(
        "IBM trial accuracy: "
        f"mean={trial_acc.mean():.3f}, std={trial_acc.std():.3f}, "
        f"perfect={(trial_acc == 1.0).sum()}/{N_TRIALS}"
    )


if __name__ == "__main__":
    main()
