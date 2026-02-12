"""Task 4 - Learn the 8-bit memory task with an IBM reservoir.

This script follows the same workflow as the other example tasks:
1. Build and train on the full 8-bit dataset.
2. Evaluate on random challenges.
3. Save a histogram plus a per-bit correctness heatmap.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from computingMicrobiome.benchmarks.k_bit_memory_bm import (
    build_dataset_output_window_only,
    evaluate_memory_trials,
)
from computingMicrobiome.ibm import make_ibm_config_from_species
from computingMicrobiome.readouts.factory import make_readout

# Parameters
BITS = 8
BOUNDARY = "periodic"
RECURRENCE = 4
ITR = 12
D_PERIOD = 8
SEED_TRAIN = 0
SEED_TRIALS = 42
N_CHALLENGES = 100

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_4_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# Delay in simulation steps from bit injection to output window.
# For this benchmark, each tick is separated by (ITR + 1) steps.
TRACE_DEPTH = (D_PERIOD + BITS) * (ITR + 1) + 8

IBM_CFG = make_ibm_config_from_species(
    # Use a small subset of the global IBM universe for this task.
    species_indices=[0, 1, 2],
    height=8,
    width_grid=8,
    # Keep reservoir-backend specific settings aligned with this benchmark.
    overrides={
        "state_width_mode": "raw",
        "input_trace_depth": TRACE_DEPTH,
        "input_trace_channels": 4,
        "input_trace_decay": 1.0,
        # For this benchmark we do not inject additional resources from input.
        "inject_scale": 0.0,
        "dilution_p": 0.0,
        "diff_numer": 0,
    },
)


def main() -> None:
    width = int(IBM_CFG["height"]) * int(IBM_CFG["width_grid"])

    print(
        f"Training SVM on all {2**BITS} possible {BITS}-bit patterns "
        "with IBM reservoir..."
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
        reservoir_kind="ibm",
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
        reservoir_kind="ibm",
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

    # Histogram (same style as tasks 1-3)
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, BITS + 2)
    ax_hist.hist(accuracies, bins=bins, edgecolor="white", color="#4C72B0", alpha=0.85)
    ax_hist.set_xlabel("Fraction of bits correctly recalled", fontsize=12)
    ax_hist.set_ylabel("Number of challenges", fontsize=12)
    ax_hist.set_title(
        f"8-bit Memory Task - IBM Reservoir + SVM\n"
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

    hist_path = OUT_DIR / "task_4_accuracy_histogram.png"
    fig_hist.savefig(hist_path, dpi=150)
    print(f"Histogram saved to {hist_path}")

    # Per-trial/per-bit correctness heatmap
    fig_heat, ax_heat = plt.subplots(figsize=(7, 8))
    cmap = ListedColormap(["#d62728", "#2ca02c"])  # wrong / correct
    norm = BoundaryNorm([-1.5, 0.0, 1.5], cmap.N)
    ax_heat.imshow(correctness, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax_heat.set_title("IBM trial-bit correctness heatmap", fontsize=12)
    ax_heat.set_xlabel("Bit index")
    ax_heat.set_ylabel("Challenge index")
    ax_heat.set_xticks(np.arange(BITS))
    ax_heat.set_yticks(np.arange(0, N_CHALLENGES, max(1, N_CHALLENGES // 10)))
    ax_heat.legend(
        handles=[
            Patch(color="#2ca02c", label="correct"),
            Patch(color="#d62728", label="wrong"),
        ],
        loc="upper right",
        framealpha=0.95,
    )
    fig_heat.tight_layout()

    heatmap_path = OUT_DIR / "task_4_trial_bit_heatmap.png"
    fig_heat.savefig(heatmap_path, dpi=150)
    print(f"Heatmap saved to {heatmap_path}")

    plt.show()


if __name__ == "__main__":
    main()
