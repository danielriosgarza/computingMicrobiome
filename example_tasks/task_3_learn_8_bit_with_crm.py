"""Task 3 – Learn the 8-bit memory task using a CRM reservoir.

Uses the benchmark-winning CRM configuration.  First reproduces the 95 %
baseline, then tests with code improvements to push toward 100 %.
"""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from computingMicrobiome.benchmarks.k_bit_memory_bm import (
    build_dataset_output_window_only,
    evaluate_memory_trials,
    train_memory_readout,
)
from computingMicrobiome.readouts.factory import make_readout

# ── Parameters ────────────────────────────────────────────────────────────────
BITS = 8
BOUNDARY = "periodic"
RECURRENCE = 4
ITR = 12
D_PERIOD = 8
N_CHALLENGES = 100

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_3_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# Exact benchmark-winning CRM config (trial 5, 95 % baseline)
_BASE_CRM = {
    "height": 18,
    "width_grid": 14,
    "n_species": 4,
    "n_resources": 2,
    "reaction_matrix": [
        [0.36859479546546936, 0.380249559879303],
        [0.4740954041481018, 0.4701269865036011],
        [0.13425308465957642, 0.44110798835754395],
        [0.3227876126766205, 0.284614622592926],
    ],
    "consumption_matrix": [
        [0.3911444842815399, 0.04048113897442818],
        [0.3331524729728699, 0.16138195991516113],
        [0.22661276161670685, 0.041711173951625824],
        [0.16788119077682495, 0.14139164984226227],
    ],
    "resource_inflow": [0.07948277145624161, 0.16246742010116577],
    "diffusion_species": [
        0.05990222841501236, 0.058150049299001694,
        0.0022665599826723337, 0.06196729093790054,
    ],
    "diffusion_resources": [0.0361427403986454, 0.049262188374996185],
    "dt": 0.06601391479639832,
    "dilution": 0.03028053845976663,
    "noise_std": 0.005763613344431059,
    "inject_scale": 0.05453666153993868,
    "projection": {"kind": "random", "output_width": 320, "seed": 10005, "scale": 1.0},
}

WIDTH = 18 * 14  # 252


def _make_config(**overrides) -> dict:
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in _BASE_CRM.items()}
    cfg.update(overrides)
    return cfg


def _run_variant(label: str, crm_cfg: dict, seed: int):
    """Build dataset, fit SVM, evaluate 100 challenges.  Returns accuracies."""
    t0 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"  {label}  (seed={seed})")
    print(f"{'='*60}")

    X, y, input_locations = build_dataset_output_window_only(
        bits=BITS,
        rule_number=110,
        width=WIDTH,
        boundary=BOUNDARY,
        recurrence=RECURRENCE,
        itr=ITR,
        d_period=D_PERIOD,
        seed=seed,
        reservoir_kind="crm",
        reservoir_config=crm_cfg,
    )
    rng = np.random.default_rng(seed)
    reg = make_readout("svm", {"C": 20.0, "class_weight": "balanced"}, rng=rng)
    reg.fit(X, y)
    train_acc = reg.score(X, y)
    print(f"  Training shape: {X.shape}   Training accuracy: {train_acc:.4f}")

    correctness = evaluate_memory_trials(
        reg=reg, bits=BITS, rule_number=110, width=WIDTH, boundary=BOUNDARY,
        recurrence=RECURRENCE, itr=ITR, d_period=D_PERIOD,
        input_locations=input_locations, n_trials=N_CHALLENGES,
        seed_trials=42, reservoir_kind="crm", reservoir_config=crm_cfg,
    )
    accs = (correctness == 1).mean(axis=1)
    elapsed = time.perf_counter() - t0
    print(f"  Eval mean: {accs.mean():.3f}  std: {accs.std():.3f}  "
          f"perfect: {(accs==1).sum()}/{N_CHALLENGES}  ({elapsed:.0f}s)")
    return train_acc, accs


# ── Run experiments ───────────────────────────────────────────────────────────
results = {}

# A) Exact benchmark reproduction (original CRM with noise — overfits)
results["baseline_noise"] = _run_variant(
    "A – Baseline (noise=0.006)",
    _make_config(), seed=1503827930,
)

# B) Just remove noise (deterministic → reproducible features)
results["no_noise"] = _run_variant(
    "B – noise_std=0 only",
    _make_config(noise_std=0.0), seed=1503827930,
)

# C) noise=0 + all improvements
results["improved"] = _run_variant(
    "C – All improvements",
    _make_config(
        noise_std=0.0,
        inject_mode="both",
        half_saturation=1.5,
        normalize_state=True,
        cross_features=False,
        basal_init=True,
        basal_species=0.1,
    ),
    seed=1503827930,
)

# ── Summary plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
bins = np.linspace(0, 1, BITS + 2)
labels = ["A – Baseline\n(noise=0.006)", "B – noise=0\nonly", "C – All\nimprovements"]

for ax, (key, (train_acc, accs)), lab in zip(axes, results.items(), labels):
    ax.hist(accs, bins=bins, edgecolor="white", color="#4C72B0", alpha=0.85)
    ax.axvline(accs.mean(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"mean={accs.mean():.2f}")
    ax.set_title(f"{lab}\ntrain={train_acc:.2f}", fontsize=10)
    ax.set_xticks(np.arange(0, BITS + 1) / BITS)
    ax.set_xticklabels([f"{i}/{BITS}" for i in range(BITS + 1)], fontsize=8)
    ax.set_xlim(-0.02, 1.05)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Number of challenges", fontsize=11)
fig.suptitle("8-bit Memory Task – CRM Reservoir + SVM (benchmark-winning matrices)",
             fontsize=12, y=1.02)
fig.tight_layout()

out_path = OUT_DIR / "task_3_accuracy_histogram.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nHistogram saved to {out_path}")
plt.show()
