"""Task 5 - Learn the 4-bit opcode logic task using Rule 110.

Train a readout on all 64 opcode/operand combinations, report performance,
and save both a histogram and a red/green correctness heatmap.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from computingMicrobiome.benchmarks.k_opcode_logic16_bm import apply_opcode
from computingMicrobiome.models.k_opcode_logic16 import KOpcodeLogic16
from computingMicrobiome.plot_utils import plot_red_green_grid

OPS = {
    0: "FALSE",
    1: "NOR",
    2: "A'B",
    3: "NOT A",
    4: "A B'",
    5: "NOT B",
    6: "XOR",
    7: "NAND",
    8: "AND",
    9: "XNOR",
    10: "B",
    11: "A' OR B",
    12: "A",
    13: "A OR B'",
    14: "OR",
    15: "TRUE",
}

# Parameters
RULE_NUMBER = 110
WIDTH = 256
BOUNDARY = "periodic"
RECURRENCE = 8
ITR = 8
D_PERIOD = 200
REPEATS = 1
FEATURE_MODE = "cue_tick"
OUTPUT_WINDOW = 2
SEED_TRAIN = 0

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_5_artifacts"
OUT_DIR.mkdir(exist_ok=True)


def bits4(op: int) -> list[int]:
    return [(op >> 3) & 1, (op >> 2) & 1, (op >> 1) & 1, op & 1]


def main() -> None:
    print(
        "Training Rule 110 model for logic16 task "
        f"(width={WIDTH}, d_period={D_PERIOD}) ..."
    )
    model = KOpcodeLogic16(
        rule_number=RULE_NUMBER,
        width=WIDTH,
        boundary=BOUNDARY,
        recurrence=RECURRENCE,
        itr=ITR,
        d_period=D_PERIOD,
        repeats=REPEATS,
        feature_mode=FEATURE_MODE,
        output_window=OUTPUT_WINDOW,
        seed=SEED_TRAIN,
        readout_kind="svm",
    ).fit()
    print("Training complete.\n")

    operand_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Rows: opcode 0..15, columns: (a,b) in 00,01,10,11.
    correctness = np.full((16, 4), -1, dtype=np.int8)
    rows: list[tuple[int, int, int, int, int, bool]] = []

    for op in range(16):
        op_bits = bits4(op)
        for col, (a, b) in enumerate(operand_pairs):
            pred = int(model.predict([op_bits + [a, b]])[0])
            exp = int(apply_opcode(np.array(op_bits, dtype=np.int8), a, b))
            ok = pred == exp
            correctness[op, col] = 1 if ok else -1
            rows.append((op, a, b, exp, pred, ok))

    overall_acc = float((correctness == 1).mean())
    per_opcode_acc = (correctness == 1).mean(axis=1)

    print(f"Overall accuracy: {overall_acc:.3f} ({int((correctness == 1).sum())}/64)")
    print("\nAccuracy by opcode:")
    for op in range(16):
        print(
            f"  {OPS[op]:>7}  opcode={bits4(op)}  "
            f"acc={per_opcode_acc[op]*100:.1f}% ({int((correctness[op] == 1).sum())}/4)"
        )

    mismatches = [r for r in rows if not r[-1]]
    if mismatches:
        print("\nMismatches:")
        for op, a, b, exp, pred, _ok in mismatches:
            print(
                f"  {OPS[op]:>7} opcode={bits4(op)} a={a} b={b} "
                f"expected={exp} pred={pred}"
            )
    else:
        print("\nNo mismatches.")

    # Histogram of opcode-level accuracies (16 values, each in {0, 0.25, ..., 1.0}).
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4.5))
    bins = np.array([-0.125, 0.125, 0.375, 0.625, 0.875, 1.125])
    ax_hist.hist(
        per_opcode_acc,
        bins=bins,
        edgecolor="white",
        color="#4C72B0",
        alpha=0.85,
    )
    ax_hist.set_xlabel("Per-opcode accuracy", fontsize=12)
    ax_hist.set_ylabel("Number of opcodes", fontsize=12)
    ax_hist.set_title("Logic16 Task - Rule 110 ECA + SVM", fontsize=13)
    ax_hist.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_hist.set_xlim(-0.05, 1.05)
    ax_hist.axvline(
        per_opcode_acc.mean(),
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"mean = {per_opcode_acc.mean():.2f}",
    )
    ax_hist.legend(fontsize=11)
    fig_hist.tight_layout()

    hist_path = OUT_DIR / "task_5_opcode_accuracy_histogram.png"
    fig_hist.savefig(hist_path, dpi=150)
    print(f"\nHistogram saved to {hist_path}")

    fig_heat, ax_heat = plot_red_green_grid(
        correctness,
        title="Logic16 trial grid (rows=opcode, cols=(a,b)=00,01,10,11)",
        show=False,
    )
    ax_heat.set_xlabel("operand pair (a,b)")
    ax_heat.set_ylabel("opcode")
    ax_heat.set_xticks(np.arange(4))
    ax_heat.set_xticklabels(["00", "01", "10", "11"])
    ax_heat.set_yticks(np.arange(16))
    ax_heat.set_yticklabels([str(i) for i in range(16)])
    fig_heat.tight_layout()

    heatmap_path = OUT_DIR / "task_5_opcode_operand_heatmap.png"
    fig_heat.savefig(heatmap_path, dpi=150)
    print(f"Heatmap saved to {heatmap_path}")

    plt.show()


if __name__ == "__main__":
    main()
