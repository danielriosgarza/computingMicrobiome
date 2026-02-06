from __future__ import annotations

import numpy as np

from .benchmarks.k_compound_opcode_bm import apply_compound_opcode
from .benchmarks.k_opcode_logic_bm import apply_opcode as apply_opcode_8
from .benchmarks.k_opcode_logic16_bm import apply_opcode as apply_opcode_16
from .eca import eca_rule_lkt
from .experiments.run_toy_addition_experiment import main as toy_addition_main
from .models import (
    KBitMemory,
    KCompoundOpcode,
    KOpcodeLogic,
    KOpcodeLogic16,
    KSerialAdder,
    KXOR,
)
from .plot_utils import make_spacetime_gif, plot_red_green_grid, plot_xor_classification_summary
from .utils import bits_lsb_to_int, int_to_bits, int_to_bits_lsb


def main_toy_addition() -> None:
    toy_addition_main()


def main_make_figures() -> None:
    from .plot_utils import main as figures_main

    figures_main()


def main_opcode_test() -> None:
    OPS = {
        0: "AND",
        1: "OR",
        2: "XOR",
        3: "NAND",
        4: "NOR",
        5: "XNOR",
        6: "A",
        7: "B",
    }

    def bits3(op: int):
        return [(op >> 2) & 1, (op >> 1) & 1, op & 1]

    alu = KOpcodeLogic(
        rule_number=30,
        width=256,
        boundary="periodic",
        recurrence=8,
        itr=8,
        d_period=20,
        repeats=1,
        feature_mode="cue_tick",
        output_window=2,
        seed=0,
    ).fit()

    print("AND(1,1) expected 1 ->", alu.predict([[0, 0, 0, 1, 1]]))
    print("NAND(1,1) expected 0 ->", alu.predict([[0, 1, 1, 1, 1]]))
    print("")

    rows = []
    for op in range(8):
        op_bits = bits3(op)
        for a in (0, 1):
            for b in (0, 1):
                X = [op_bits + [a, b]]
                pred = int(alu.predict(X)[0])
                exp = int(apply_opcode_8(np.array(op_bits, dtype=np.int8), a, b))
                ok = pred == exp
                rows.append((op, op_bits, a, b, exp, pred, ok))

    overall_acc = np.mean([r[-1] for r in rows])
    print(
        f"Overall accuracy: {overall_acc*100:.1f}% ({sum(r[-1] for r in rows)}/{len(rows)})"
    )

    print("\nAccuracy by opcode:")
    for op in range(8):
        sub = [r for r in rows if r[0] == op]
        acc = np.mean([r[-1] for r in sub])
        print(
            f"  {OPS[op]:>4}  opcode={bits3(op)}  acc={acc*100:.1f}% ({sum(r[-1] for r in sub)}/{len(sub)})"
        )

    mismatches = [r for r in rows if not r[-1]]
    if mismatches:
        print("\nMismatches:")
        for op, op_bits, a, b, exp, pred, ok in mismatches:
            print(
                f"  {OPS[op]:>4} opcode={op_bits}  a={a} b={b}  expected={exp} pred={pred}"
            )
    else:
        print("\nNo mismatches ✅")


def main_opcode16_test() -> None:
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

    def bits4(op: int):
        return [(op >> 3) & 1, (op >> 2) & 1, (op >> 1) & 1, op & 1]

    alu = KOpcodeLogic16(
        rule_number=110,
        width=256,
        boundary="periodic",
        recurrence=8,
        itr=8,
        d_period=200,
        repeats=1,
        feature_mode="cue_tick",
        output_window=2,
        seed=0,
    ).fit()

    print("XOR(1,0) expected 1 ->", alu.predict([[0, 1, 1, 0, 1, 0]]))
    print("AND(1,1) expected 1 ->", alu.predict([[1, 0, 0, 0, 1, 1]]))
    print("NOT A (a=1,b=0) expected 0 ->", alu.predict([[0, 0, 1, 1, 1, 0]]))
    print("")

    rows = []
    for op in range(16):
        op_bits = bits4(op)
        for a in (0, 1):
            for b in (0, 1):
                X = [op_bits + [a, b]]
                pred = int(alu.predict(X)[0])
                exp = int(apply_opcode_16(np.array(op_bits, dtype=np.int8), a, b))
                ok = pred == exp
                rows.append((op, op_bits, a, b, exp, pred, ok))

    overall_acc = np.mean([r[-1] for r in rows])
    print(
        f"Overall accuracy: {overall_acc*100:.1f}% ({sum(r[-1] for r in rows)}/{len(rows)})"
    )

    print("\nAccuracy by opcode:")
    for op in range(16):
        sub = [r for r in rows if r[0] == op]
        acc = np.mean([r[-1] for r in sub])
        print(
            f"  {OPS.get(op, str(op)):>7}  opcode={bits4(op)}  acc={acc*100:.1f}% ({sum(r[-1] for r in sub)}/{len(sub)})"
        )

    mismatches = [r for r in rows if not r[-1]]
    if mismatches:
        print("\nMismatches:")
        for op, op_bits, a, b, exp, pred, ok in mismatches:
            print(
                f"  {OPS.get(op, str(op)):>7} opcode={op_bits}  a={a} b={b}  expected={exp} pred={pred}"
            )
    else:
        print("\nNo mismatches ✅")


def main_compound_opcode_test() -> None:
    def bits4(op: int):
        return [(op >> 3) & 1, (op >> 2) & 1, (op >> 1) & 1, op & 1]

    model = KCompoundOpcode(
        rule_number=110,
        width=256,
        boundary="periodic",
        recurrence=8,
        itr=8,
        d_period=20,
        repeats=1,
        feature_mode="cue_tick",
        output_window=2,
        seed=0,
    ).fit()

    print(
        "XOR(1,0)=1 then AND(1,1)=1 ->",
        model.predict([[0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]),
    )
    print(
        "OR(0,1)=1 then NAND(1,1)=0 ->",
        model.predict([[1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]]),
    )
    print("")

    rows = []
    for op1 in range(16):
        op1_bits = bits4(op1)
        for a in (0, 1):
            for b in (0, 1):
                for op2 in range(16):
                    op2_bits = bits4(op2)
                    for c in (0, 1):
                        X = [op1_bits + [a, b] + op2_bits + [c]]
                        pred = int(model.predict(X)[0])
                        exp = int(
                            apply_compound_opcode(
                                np.array(op1_bits, dtype=np.int8),
                                a,
                                b,
                                np.array(op2_bits, dtype=np.int8),
                                c,
                            )
                        )
                        ok = pred == exp
                        rows.append((op1, op1_bits, a, b, op2, op2_bits, c, exp, pred, ok))

    overall_acc = np.mean([r[-1] for r in rows])
    print(
        f"Overall accuracy: {overall_acc*100:.1f}% ({sum(r[-1] for r in rows)}/{len(rows)})"
    )

    print("\nAccuracy by op2:")
    for op2 in range(16):
        sub = [r for r in rows if r[4] == op2]
        acc = np.mean([r[-1] for r in sub])
        print(
            f"  op2={bits4(op2)}  acc={acc*100:.1f}% ({sum(r[-1] for r in sub)}/{len(sub)})"
        )

    mismatches = [r for r in rows if not r[-1]]
    if mismatches:
        print("\nMismatches (first 10):")
        for op1, op1_bits, a, b, op2, op2_bits, c, exp, pred, ok in mismatches[:10]:
            print(
                f"  op1={op1_bits} a={a} b={b}  op2={op2_bits} c={c}  expected={exp} pred={pred}"
            )
    else:
        print("\nNo mismatches ✅")


def main_serial_adder_test() -> None:
    bits = 8
    rng = np.random.default_rng(0)

    model = KSerialAdder(
        rule_number=110,
        width=256,
        boundary="periodic",
        recurrence=8,
        itr=8,
        d_period=20,
        bits=bits,
        n_train=500,
        seed=0,
    ).fit()

    n_test = 100
    A = rng.integers(0, 2**bits, size=n_test).tolist()
    B = rng.integers(0, 2**bits, size=n_test).tolist()

    preds = model.predict(A, B)

    true_bits = np.zeros_like(preds)
    for i, (a, b) in enumerate(zip(A, B)):
        true_bits[i] = int_to_bits_lsb(a + b, bits)

    acc_per_bit = (preds == true_bits).mean(axis=0)
    print("Accuracy per bit position (LSB->MSB):")
    for k, acc in enumerate(acc_per_bit.tolist()):
        print(f"  bit {k}: {acc*100:.1f}%")

    overall = (preds == true_bits).mean()
    print(f"\nOverall bit accuracy: {overall*100:.1f}%")

    a_edge = int("01111111", 2)
    b_edge = int("00000001", 2)
    pred_edge = model.predict([a_edge], [b_edge])[0]
    true_edge = int_to_bits_lsb(a_edge + b_edge, bits)

    print("\nCascade carry test:")
    print(f"  A = 01111111 ({a_edge})")
    print(f"  B = 00000001 ({b_edge})")
    print(f"  Expected sum bits: {true_edge.tolist()} (LSB->MSB)")
    print(f"  Predicted bits:   {pred_edge.tolist()} (LSB->MSB)")
    print(f"  Expected sum: {bits_lsb_to_int(true_edge)}")
    print(f"  Predicted sum: {bits_lsb_to_int(pred_edge)}")


def main_kbit_memory_test() -> None:
    bits = 8
    rule_number = 110
    width = 700
    boundary = "periodic"
    recurrence = 4
    itr = 2
    d_period = 200
    n_trials = 100
    seed = 7

    model = KBitMemory(
        bits=bits,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        recurrence=recurrence,
        itr=itr,
        d_period=d_period,
        seed=0,
    )
    model.fit()

    rng = np.random.default_rng(seed)
    test_inputs_int = rng.integers(0, 2**bits, size=n_trials)
    test_inputs = np.array([int_to_bits(val, bits) for val in test_inputs_int])

    predicted_outputs = model.predict(test_inputs)

    correctness = np.where(predicted_outputs == test_inputs, 1, -1)

    plot_red_green_grid(
        correctness,
        title=f"Rule {rule_number} | {n_trials} trials × {bits} bits (memory only) - using KBitMemory classifier",
    )


def main_kxor_visual() -> None:
    bits = 8
    rule_number = 106
    width = 700
    boundary = "periodic"
    recurrence = 4
    itr = 1
    d_period = 200
    injection_interval = 1
    injection_repetitions = 1
    n_trials = 100

    model = KXOR(
        bits=bits,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        recurrence=recurrence,
        itr=itr,
        d_period=d_period,
        injection_interval=injection_interval,
        injection_repetitions=injection_repetitions,
        seed=0,
    )
    model.fit()

    test_inputs = np.array([int_to_bits(val, bits) for val in range(n_trials)])

    y_pred = model.predict(test_inputs)

    y_true = np.array([np.bitwise_xor.reduce(bits_arr) for bits_arr in test_inputs])

    accuracy = np.mean(y_pred == y_true)
    print(f"Overall XOR accuracy: {accuracy:.3f}")
    plot_xor_classification_summary(
        test_inputs,
        y_pred,
        y_true,
        title=f"Parity (k-bit XOR) Task with Rule {rule_number} ({n_trials} trials, {bits} bits)",
    )


def main_rule110_gif() -> None:
    rule = 110
    size = 201
    steps = 200
    boundary = "periodic"

    x0 = np.zeros(size, dtype=np.int8)
    x0[size // 2] = 1

    rule_table = eca_rule_lkt(rule)
    states = _run_eca(x0, rule_table, steps, boundary)

    output_path = make_spacetime_gif(states, "Figs/rule110.gif")
    print(f"Saved GIF to {output_path}")


def _run_eca(x0: np.ndarray, rule: np.ndarray, steps: int, boundary: str) -> np.ndarray:
    from .eca import eca_run

    return eca_run(x0, rule, steps, boundary)
