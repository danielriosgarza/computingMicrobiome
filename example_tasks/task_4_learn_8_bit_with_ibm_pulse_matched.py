"""Task 4 - Learn the 8-bit memory task with IBM pulse injection (matched setup).

This script is intentionally matched to `task_4_learn_8_bit_with_ibm.py`:
- same task schedule (BITS, RECURRENCE, ITR, D_PERIOD),
- same grid dimensions and species subset,
- same IBM core behavior via `reservoir_kind="ibm_pulse"` (reset/state-width/
  trace now matches `ibm`), plus the same feature width mode and input trace setup.

Only the injection behavior changes:
- `inject_mode="pulse_bit"` (clear patch + toxin/popular pulse).
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from computingMicrobiome.benchmarks.k_bit_memory_bm import (
    build_dataset_output_window_only,
    evaluate_memory_trials,
)
from computingMicrobiome.ibm import (
    make_channel_to_resource_from_config,
    make_ibm_config_from_species,
)
from computingMicrobiome.plot_utils import plot_red_green_grid
from computingMicrobiome.readouts.factory import make_readout

# Parameters (kept identical to task_4_learn_8_bit_with_ibm.py)
BITS = 8
BOUNDARY = "periodic"
RECURRENCE = 4
ITR = 12
D_PERIOD = 8
SEED_TRAIN = 0
SEED_TRIALS = 42
N_CHALLENGES = 100

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_4_pulse_matched_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# Delay in simulation steps from bit injection to output window.
# For this benchmark, each tick is separated by (ITR + 1) steps.
TRACE_DEPTH = (D_PERIOD + BITS) * (ITR + 1) + 8

# Number of input channels for the 8-bit memory benchmark.
N_CHANNELS = 4

# IBM reservoir dynamics (kept matched to task_4_learn_8_bit_with_ibm.py).
IBM_DIFF_NUMER = 1
IBM_DILUTION_P = 0.02
IBM_INJECT_SCALE = 2.0

# Pulse parameters (only injection behavior differs).
PULSE_RADIUS = 2
PULSE_TOXIN_CONC = 180
PULSE_POPULAR_CONC = 200

IBM_CFG = make_ibm_config_from_species(
    species_indices=[0, 1, 2],
    height=8,
    width_grid=8,
    overrides={
        "state_width_mode": "raw",
        "input_trace_depth": TRACE_DEPTH,
        "input_trace_channels": N_CHANNELS,
        "input_trace_decay": 1.0,
        "inject_scale": IBM_INJECT_SCALE,
        "dilution_p": IBM_DILUTION_P,
        "diff_numer": IBM_DIFF_NUMER,
        "inject_mode": "pulse_bit",
        "pulse_radius": PULSE_RADIUS,
        "pulse_toxin_conc": PULSE_TOXIN_CONC,
        "pulse_popular_conc": PULSE_POPULAR_CONC,
    },
)
# Keep channel mapping identical to task_4 for schedule/channel parity.
IBM_CFG["channel_to_resource"] = make_channel_to_resource_from_config(
    IBM_CFG, N_CHANNELS
)


def main() -> None:
    width = int(IBM_CFG["height"]) * int(IBM_CFG["width_grid"])

    print(
        f"Training SVM on all {2**BITS} possible {BITS}-bit patterns "
        "with IBM pulse-injection reservoir (matched task-4 setup)..."
    )
    X, y, input_locations = build_dataset_output_window_only(
        bits=BITS,
        rule_number=110,
        width=width,
        boundary=BOUNDARY,
        recurrence=RECURRENCE,
        itr=ITR,
        d_period=D_PERIOD,
        seed=SEED_TRAIN,
        reservoir_kind="ibm_pulse",
        reservoir_config=IBM_CFG,
    )
    rng = np.random.default_rng(SEED_TRAIN)
    reg = make_readout("svm", {"C": 10.0, "class_weight": "balanced"}, rng=rng)
    reg.fit(X, y)
    train_acc = float(reg.score(X, y))
    print("Training complete.\n")

    print(
        f"Evaluating on {N_CHALLENGES} random challenges (seed={SEED_TRIALS})..."
    )
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
        n_trials=N_CHALLENGES,
        seed_trials=SEED_TRIALS,
        reservoir_kind="ibm_pulse",
        reservoir_config=IBM_CFG,
    )
    accuracies = (correctness == 1).mean(axis=1)

    print(f"  Training shape : X={X.shape}, y={y.shape}")
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Mean accuracy  : {accuracies.mean():.3f}")
    print(f"  Std  accuracy  : {accuracies.std():.3f}")
    print(f"  Min  accuracy  : {accuracies.min():.3f}")
    print(f"  Max  accuracy  : {accuracies.max():.3f}")
    print(f"  Perfect trials : {(accuracies == 1.0).sum()} / {N_CHALLENGES}\n")

    # Histogram (same style as tasks 1-3).
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, BITS + 2)
    ax_hist.hist(accuracies, bins=bins, edgecolor="white", color="#4C72B0", alpha=0.85)
    ax_hist.set_xlabel("Fraction of bits correctly recalled", fontsize=12)
    ax_hist.set_ylabel("Number of challenges", fontsize=12)
    ax_hist.set_title(
        f"8-bit Memory Task - IBM Reservoir (pulse injection) + SVM\n"
        f"({N_CHALLENGES} challenges, width={width}, d_period={D_PERIOD})",
        fontsize=13,
    )
    ax_hist.set_xticks(np.arange(0, BITS + 1) / BITS)
    ax_hist.set_xticklabels([f"{i}/{BITS}" for i in range(BITS + 1)])
    ax_hist.set_xlim(-0.02, 1.05)
    ax_hist.axvline(
        accuracies.mean(),
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"mean = {accuracies.mean():.2f}",
    )
    ax_hist.legend(fontsize=11)
    fig_hist.tight_layout()

    hist_path = OUT_DIR / "task_4_pulse_matched_accuracy_histogram.png"
    fig_hist.savefig(hist_path, dpi=150)
    print(f"Histogram saved to {hist_path}")

    # Per-trial/per-bit correctness heatmap.
    fig_heat, _ = plot_red_green_grid(
        correctness,
        title="IBM pulse-matched trial-bit correctness heatmap",
        show=False,
    )

    heatmap_path = OUT_DIR / "task_4_pulse_matched_trial_bit_heatmap.png"
    fig_heat.savefig(heatmap_path, dpi=150)
    print(f"Heatmap saved to {heatmap_path}")

    plt.show()


if __name__ == "__main__":
    main()
