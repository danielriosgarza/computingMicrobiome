from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.svm import SVC

from ..benchmarks.k_opcode_logic16_bm import apply_opcode, run_episode_record_tagged
from ..utils import bits_lsb_to_int, create_input_locations, int_to_bits_lsb

OP_XOR = np.array([0, 1, 1, 0], dtype=np.int8)
OP_AND = np.array([1, 0, 0, 0], dtype=np.int8)
OP_OR = np.array([1, 1, 1, 0], dtype=np.int8)


def int_to_bits(x: int, n: int) -> List[int]:
    """LSB-first bits of length n."""
    return int_to_bits_lsb(x, n).astype(int).tolist()


def bits_to_int(bits: Sequence[int]) -> int:
    """Convert LSB-first bits to integer."""
    return bits_lsb_to_int(np.asarray(bits, dtype=np.int8))


def enumerate_addition_dataset(
    n_bits: int, cin: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Exhaustively enumerate addition dataset."""
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    cin = int(cin) & 1

    n_samples = 2 ** (2 * n_bits)
    X_direct = np.zeros((n_samples, 2 * n_bits), dtype=np.int8)
    Y = np.zeros((n_samples, n_bits + 1), dtype=np.int8)

    idx = 0
    for x in range(2**n_bits):
        x_bits = int_to_bits(x, n_bits)
        for y in range(2**n_bits):
            y_bits = int_to_bits(y, n_bits)
            X_direct[idx] = np.array(x_bits + y_bits, dtype=np.int8)

            s = x + y + cin
            sum_bits = int_to_bits(s, n_bits)
            cout = (s >> n_bits) & 1
            Y[idx] = np.array(sum_bits + [cout], dtype=np.int8)
            idx += 1

    return X_direct, Y


def train_direct_linear_models(X_direct: np.ndarray, Y: np.ndarray) -> List[SVC]:
    """Train one linear SVM per output bit."""
    X_direct = np.asarray(X_direct, dtype=np.int8)
    Y = np.asarray(Y, dtype=np.int8)
    models: List[SVC] = []
    for i in range(Y.shape[1]):
        reg = SVC(kernel="linear")
        reg.fit(X_direct, Y[:, i])
        models.append(reg)
    return models


def evaluate_models(models: List[SVC], X: np.ndarray, Y: np.ndarray) -> dict:
    """Evaluate per-bit and full-vector accuracy."""
    X = np.asarray(X, dtype=np.int8)
    Y = np.asarray(Y, dtype=np.int8)

    preds = np.zeros_like(Y, dtype=np.int8)
    for i, reg in enumerate(models):
        preds[:, i] = reg.predict(X).astype(np.int8)

    per_bit = (preds == Y).mean(axis=0).tolist()
    full_vec = (preds == Y).all(axis=1)
    full_acc = float(full_vec.mean())

    return {
        "per_bit": per_bit,
        "full_acc": full_acc,
        "full_correct": int(full_vec.sum()),
        "n_samples": int(Y.shape[0]),
    }


def _full_adder_reservoir_features_with_locations(
    a: int,
    b: int,
    cin: int,
    *,
    rule_number: int,
    width: int,
    boundary: str,
    itr: int,
    d_period: int,
    recurrence: int,
    repeats: int,
    rng: np.random.Generator,
    input_locations: np.ndarray,
    feature_mode: str = "cue_tick",
    output_window: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reservoir features and labels for one full-adder micro-episode."""
    a = int(a) & 1
    b = int(b) & 1
    cin = int(cin) & 1

    feats: List[np.ndarray] = []

    # step0: t = XOR(a,b)
    ep0 = run_episode_record_tagged(
        op_bits_msb_first=OP_XOR,
        a=a,
        b=b,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        itr=itr,
        d_period=d_period,
        rng=rng,
        input_locations=input_locations,
        repeats=repeats,
        order=None,
        reg=None,
        collect_states=False,
        x0_mode="zeros",
        feature_mode=feature_mode,
        output_window=output_window,
    )
    t = apply_opcode(OP_XOR, a, b)
    feats.append(ep0["X_episode"].reshape(-1))

    # step1: sum = XOR(t, cin)
    ep1 = run_episode_record_tagged(
        op_bits_msb_first=OP_XOR,
        a=t,
        b=cin,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        itr=itr,
        d_period=d_period,
        rng=rng,
        input_locations=input_locations,
        repeats=repeats,
        order=None,
        reg=None,
        collect_states=False,
        x0_mode="zeros",
        feature_mode=feature_mode,
        output_window=output_window,
    )
    sum_bit = apply_opcode(OP_XOR, t, cin)
    feats.append(ep1["X_episode"].reshape(-1))

    # step2: u = AND(cin, t)
    ep2 = run_episode_record_tagged(
        op_bits_msb_first=OP_AND,
        a=cin,
        b=t,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        itr=itr,
        d_period=d_period,
        rng=rng,
        input_locations=input_locations,
        repeats=repeats,
        order=None,
        reg=None,
        collect_states=False,
        x0_mode="zeros",
        feature_mode=feature_mode,
        output_window=output_window,
    )
    u = apply_opcode(OP_AND, cin, t)
    feats.append(ep2["X_episode"].reshape(-1))

    # step3: v = AND(a, b)
    ep3 = run_episode_record_tagged(
        op_bits_msb_first=OP_AND,
        a=a,
        b=b,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        itr=itr,
        d_period=d_period,
        rng=rng,
        input_locations=input_locations,
        repeats=repeats,
        order=None,
        reg=None,
        collect_states=False,
        x0_mode="zeros",
        feature_mode=feature_mode,
        output_window=output_window,
    )
    v = apply_opcode(OP_AND, a, b)
    feats.append(ep3["X_episode"].reshape(-1))

    # step4: cout = OR(u, v)
    ep4 = run_episode_record_tagged(
        op_bits_msb_first=OP_OR,
        a=u,
        b=v,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        itr=itr,
        d_period=d_period,
        rng=rng,
        input_locations=input_locations,
        repeats=repeats,
        order=None,
        reg=None,
        collect_states=False,
        x0_mode="zeros",
        feature_mode=feature_mode,
        output_window=output_window,
    )
    cout = apply_opcode(OP_OR, u, v)
    feats.append(ep4["X_episode"].reshape(-1))

    F = np.concatenate(feats, axis=0)
    y = np.array([sum_bit, cout], dtype=np.int8)
    return F, y


def full_adder_reservoir_features(
    a: int,
    b: int,
    cin: int,
    *,
    rule_number: int,
    width: int,
    boundary: str,
    itr: int,
    d_period: int,
    recurrence: int,
    repeats: int,
    seed: int,
    feature_mode: str = "cue_tick",
    output_window: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Reservoir features and labels for a single full-adder step."""
    rng = np.random.default_rng(seed)
    input_locations = create_input_locations(width, recurrence, 10, rng)

    F, y = _full_adder_reservoir_features_with_locations(
        a,
        b,
        cin,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        itr=itr,
        d_period=d_period,
        recurrence=recurrence,
        repeats=repeats,
        rng=rng,
        input_locations=input_locations,
        feature_mode=feature_mode,
        output_window=output_window,
    )
    return F, y, int(y[0]), int(y[1])


def addition_reservoir_features(
    x: int,
    y: int,
    n_bits: int,
    cin: int,
    **reservoir_params,
) -> Tuple[np.ndarray, np.ndarray]:
    """N-bit ripple carry reservoir features (LSB-first)."""
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    cin = int(cin) & 1

    rule_number = int(reservoir_params["rule_number"])
    width = int(reservoir_params["width"])
    boundary = str(reservoir_params["boundary"])
    itr = int(reservoir_params["itr"])
    d_period = int(reservoir_params["d_period"])
    recurrence = int(reservoir_params["recurrence"])
    repeats = int(reservoir_params.get("repeats", 1))
    seed = int(reservoir_params.get("seed", 0))
    feature_mode = str(reservoir_params.get("feature_mode", "cue_tick"))
    output_window = int(reservoir_params.get("output_window", 2))

    rng = np.random.default_rng(seed)
    input_locations = create_input_locations(width, recurrence, 10, rng)

    x_bits = int_to_bits(x, n_bits)
    y_bits = int_to_bits(y, n_bits)

    carry = cin
    features: List[np.ndarray] = []
    sum_bits: List[int] = []

    for i in range(n_bits):
        Fi, yi = _full_adder_reservoir_features_with_locations(
            x_bits[i],
            y_bits[i],
            carry,
            rule_number=rule_number,
            width=width,
            boundary=boundary,
            itr=itr,
            d_period=d_period,
            recurrence=recurrence,
            repeats=repeats,
            rng=rng,
            input_locations=input_locations,
            feature_mode=feature_mode,
            output_window=output_window,
        )
        sum_bits.append(int(yi[0]))
        carry = int(yi[1])
        features.append(Fi)

    Phi = np.concatenate(features, axis=0)
    Y_episode = np.array(sum_bits + [carry], dtype=np.int8)
    return Phi, Y_episode


def build_reservoir_dataset(
    n_bits: int,
    cin: int,
    **reservoir_params,
) -> Tuple[np.ndarray, np.ndarray]:
    """Enumerate all x,y pairs and build reservoir features."""
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    cin = int(cin) & 1

    X_res: List[np.ndarray] = []
    Y: List[np.ndarray] = []

    for x in range(2**n_bits):
        for y in range(2**n_bits):
            Phi, Y_episode = addition_reservoir_features(
                x,
                y,
                n_bits,
                cin,
                **reservoir_params,
            )
            X_res.append(Phi)
            Y.append(Y_episode)

    X = np.vstack(X_res).astype(np.int8)
    Y = np.vstack(Y).astype(np.int8)
    return X, Y


def train_reservoir_linear_models(X_res: np.ndarray, Y: np.ndarray) -> List[SVC]:
    """Train one linear SVM per output bit on reservoir features."""
    X_res = np.asarray(X_res, dtype=np.int8)
    Y = np.asarray(Y, dtype=np.int8)
    models: List[SVC] = []
    for i in range(Y.shape[1]):
        reg = SVC(kernel="linear")
        reg.fit(X_res, Y[:, i])
        models.append(reg)
    return models


def bit_balance(Y: np.ndarray) -> np.ndarray:
    """
    Return p(y=1) for each output bit.
    Y shape: (n_samples, n_bits_out)
    """
    Y = np.asarray(Y, dtype=np.int8)
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (n_samples, n_bits_out).")
    return Y.mean(axis=0)


def majority_baseline_accuracy(Y: np.ndarray) -> np.ndarray:
    """
    For each output bit, compute accuracy of predicting the majority class.
    Returns array shape (n_bits_out,).
    """
    p1 = bit_balance(Y)
    p0 = 1.0 - p1
    return np.maximum(p0, p1)


def constant_zero_baseline_accuracy(Y: np.ndarray) -> np.ndarray:
    """
    Accuracy per bit if always predicting 0.
    """
    p1 = bit_balance(Y)
    return 1.0 - p1


def evaluate_linear_task(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Trains linear SVMs per output bit and returns:
      - per_bit_acc
      - full_acc
      - full_correct, n_samples
      - p1 (bit balance)
      - majority_baseline_per_bit
      - zero_baseline_per_bit
    """
    X = np.asarray(X, dtype=np.int8)
    Y = np.asarray(Y, dtype=np.int8)

    models: List[SVC] = []
    for i in range(Y.shape[1]):
        reg = SVC(kernel="linear")
        reg.fit(X, Y[:, i])
        models.append(reg)

    eval_res = evaluate_models(models, X, Y)
    p1 = bit_balance(Y)
    majority = majority_baseline_accuracy(Y)
    zero_base = constant_zero_baseline_accuracy(Y)

    return {
        "per_bit_acc": eval_res["per_bit"],
        "full_acc": eval_res["full_acc"],
        "full_correct": eval_res["full_correct"],
        "n_samples": eval_res["n_samples"],
        "p1": p1.tolist(),
        "majority_baseline_per_bit": majority.tolist(),
        "zero_baseline_per_bit": zero_base.tolist(),
    }
