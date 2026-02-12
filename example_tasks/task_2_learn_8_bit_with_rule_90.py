"""Task 2 – Learn the 8-bit memory task using Rule 90.

Train a linear SVM readout on *all* 256 possible 8-bit patterns fed through
a Rule-90 ECA reservoir, then evaluate recall on 100 random challenges and
plot a histogram of per-challenge accuracy.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from computingMicrobiome.benchmarks.k_bit_memory_bm import evaluate_memory_trials
from computingMicrobiome.models.k_bit_memory import KBitMemory

# ── Parameters ────────────────────────────────────────────────────────────────
BITS = 8
RULE_NUMBER = 90
WIDTH = 700
BOUNDARY = "periodic"
RECURRENCE = 4
ITR = 2
D_PERIOD = 200
SEED_TRAIN = 0
SEED_TRIALS = 42
N_CHALLENGES = 100

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_2_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Train SVM on all 2^8 = 256 possible cases ─────────────────────────────
print(f"Training SVM on all {2**BITS} possible {BITS}-bit patterns "
      f"with Rule {RULE_NUMBER} ECA reservoir …")

model = KBitMemory(
    bits=BITS,
    rule_number=RULE_NUMBER,
    width=WIDTH,
    boundary=BOUNDARY,
    recurrence=RECURRENCE,
    itr=ITR,
    d_period=D_PERIOD,
    seed=SEED_TRAIN,
    readout_kind="svm",
)
model.fit()

print("Training complete.\n")

# ── 2. Evaluate on 100 random challenges ──────────────────────────────────────
print(f"Evaluating on {N_CHALLENGES} random challenges (seed={SEED_TRIALS}) …")

correctness = evaluate_memory_trials(
    reg=model.reg_,
    bits=BITS,
    rule_number=RULE_NUMBER,
    width=WIDTH,
    boundary=BOUNDARY,
    recurrence=RECURRENCE,
    itr=ITR,
    d_period=D_PERIOD,
    input_locations=model.input_locations_,
    n_trials=N_CHALLENGES,
    seed_trials=SEED_TRIALS,
)

# Per-challenge accuracy: fraction of bits correctly recalled
# correctness has shape (N_CHALLENGES, BITS); values ∈ {-1, 1}
accuracies = (correctness == 1).mean(axis=1)  # shape (N_CHALLENGES,)

print(f"  Mean accuracy : {accuracies.mean():.3f}")
print(f"  Std  accuracy : {accuracies.std():.3f}")
print(f"  Min  accuracy : {accuracies.min():.3f}")
print(f"  Max  accuracy : {accuracies.max():.3f}")
print(f"  Perfect trials: {(accuracies == 1.0).sum()} / {N_CHALLENGES}\n")

# ── 3. Plot histogram ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

bins = np.linspace(0, 1, BITS + 2)  # one bin per possible fraction 0/8 … 8/8
ax.hist(accuracies, bins=bins, edgecolor="white", color="#4C72B0", alpha=0.85)

ax.set_xlabel("Fraction of bits correctly recalled", fontsize=12)
ax.set_ylabel("Number of challenges", fontsize=12)
ax.set_title(
    f"8-bit Memory Task – Rule {RULE_NUMBER} ECA + SVM\n"
    f"({N_CHALLENGES} challenges, width={WIDTH}, d_period={D_PERIOD})",
    fontsize=13,
)
ax.set_xticks(np.arange(0, BITS + 1) / BITS)
ax.set_xticklabels([f"{i}/{BITS}" for i in range(BITS + 1)])
ax.set_xlim(-0.02, 1.05)

# Annotate mean
ax.axvline(accuracies.mean(), color="crimson", linestyle="--", linewidth=1.5,
           label=f"mean = {accuracies.mean():.2f}")
ax.legend(fontsize=11)

fig.tight_layout()

out_path = OUT_DIR / "task_2_accuracy_histogram.png"
fig.savefig(out_path, dpi=150)
print(f"Histogram saved to {out_path}")
plt.show()
