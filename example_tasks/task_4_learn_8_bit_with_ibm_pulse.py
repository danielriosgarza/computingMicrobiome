"""Task 4 - 8-bit memory with IBM reservoir (pulse injection, notebook setup).

Mirrors the notebook's Task 4: CROSS_FEED_6_SPECIES, 8x32 grid, left source column,
and pulse injection (bit 0 = toxin, bit 1 = popular metabolite) for the first 8 ticks.
Same workflow as task_4_learn_8_bit_with_ibm.py but uses reservoir_kind="ibm_pulse"
so you can compare performance.
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
    CROSS_FEED_6_SPECIES,
    make_ibm_config_from_species,
)
from computingMicrobiome.plot_utils import plot_red_green_grid
from computingMicrobiome.readouts.factory import make_readout

# Same task parameters as task_4_learn_8_bit_with_ibm.py
BITS = 8
BOUNDARY = "periodic"
RECURRENCE = 4
ITR = 12
D_PERIOD = 8
SEED_TRAIN = 0
SEED_TRIALS = 42
N_CHALLENGES = 100

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_4_pulse_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# Notebook Task 4 setup: cross-feed species, 8x32 grid, basal init, left source, pulse params
IBM_PULSE_CFG = make_ibm_config_from_species(
    species_indices=CROSS_FEED_6_SPECIES,
    height=8,
    width_grid=32,
    overrides={
        "basal_energy": 4,
        "dilution_p": 0.05,
        "basal_init": True,
        "pulse_radius": 2,
        "pulse_toxin_conc": 180,
        "pulse_popular_conc": 200,
        "left_source_outcompete_margin": 1,
        "left_source_colonize_empty": True,
    },
)
# Left source: one species per row cycling (default in backend)
# Pulse center: (H//2, W//2) by default in backend


def main() -> None:
    width = int(IBM_PULSE_CFG["height"]) * int(IBM_PULSE_CFG["width_grid"])

    print(
        f"Training SVM on all {2**BITS} possible {BITS}-bit patterns "
        "with IBM pulse reservoir (notebook setup)..."
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
        reservoir_config=IBM_PULSE_CFG,
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
        reservoir_config=IBM_PULSE_CFG,
    )
    accuracies = (correctness == 1).mean(axis=1)

    print(f"  Training shape : X={X.shape}, y={y.shape}")
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Mean accuracy  : {accuracies.mean():.3f}")
    print(f"  Std  accuracy  : {accuracies.std():.3f}")
    print(f"  Min  accuracy  : {accuracies.min():.3f}")
    print(f"  Max  accuracy  : {accuracies.max():.3f}")
    print(f"  Perfect trials : {(accuracies == 1.0).sum()} / {N_CHALLENGES}\n")

    # Histogram
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, BITS + 2)
    ax_hist.hist(accuracies, bins=bins, edgecolor="white", color="#4C72B0", alpha=0.85)
    ax_hist.set_xlabel("Fraction of bits correctly recalled", fontsize=12)
    ax_hist.set_ylabel("Number of challenges", fontsize=12)
    ax_hist.set_title(
        f"8-bit Memory Task - IBM Pulse Reservoir (notebook setup) + SVM\n"
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

    hist_path = OUT_DIR / "task_4_pulse_accuracy_histogram.png"
    fig_hist.savefig(hist_path, dpi=150)
    print(f"Histogram saved to {hist_path}")

    # Per-trial/per-bit correctness heatmap
    fig_heat, _ = plot_red_green_grid(
        correctness,
        title="IBM pulse trial-bit correctness heatmap",
        show=False,
    )

    heatmap_path = OUT_DIR / "task_4_pulse_trial_bit_heatmap.png"
    fig_heat.savefig(heatmap_path, dpi=150)
    print(f"Heatmap saved to {heatmap_path}")

    plt.show()


if __name__ == "__main__":
    main()
