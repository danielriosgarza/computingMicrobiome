"""CLI entry point for the toy addition experiment."""

import argparse

from experiments.toy_addition_experiment import (
    build_reservoir_dataset,
    enumerate_addition_dataset,
    evaluate_models,
    train_direct_linear_models,
    train_reservoir_linear_models,
)


def _print_results(label: str, res: dict) -> None:
    """Print experiment results to stdout.

    Args:
        label: Label for the result block.
        res: Result dictionary from evaluate_models.
    """
    per_bit = res["per_bit"]
    full_acc = res["full_acc"]
    full_correct = res["full_correct"]
    n_samples = res["n_samples"]

    print(f"{label} per-bit accuracy:", ["{:.1f}%".format(100 * p) for p in per_bit])
    print(
        f"{label} full-vector accuracy: {full_acc*100:.1f}% ({full_correct}/{n_samples})"
    )


def main() -> None:
    """Run the toy addition experiment from the command line."""
    parser = argparse.ArgumentParser(description="Toy addition experiment")
    parser.add_argument("--n_bits", type=int, default=3)
    parser.add_argument("--cin", type=int, default=0)

    parser.add_argument("--rule_number", type=int, default=110)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--boundary", type=str, default="periodic")
    parser.add_argument("--recurrence", type=int, default=8)
    parser.add_argument("--itr", type=int, default=8)
    parser.add_argument("--d_period", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_mode", type=str, default="cue_tick")
    parser.add_argument("--output_window", type=int, default=2)

    args = parser.parse_args()

    print(f"N bits: {args.n_bits}")
    print(f"Dataset size: {2 ** (2 * args.n_bits)}")

    # Direct baseline
    X_direct, Y_direct = enumerate_addition_dataset(args.n_bits, cin=args.cin)
    direct_models = train_direct_linear_models(X_direct, Y_direct)
    direct_res = evaluate_models(direct_models, X_direct, Y_direct)
    _print_results("Direct", direct_res)

    # Reservoir-mediated
    X_res, Y_res = build_reservoir_dataset(
        n_bits=args.n_bits,
        cin=args.cin,
        rule_number=args.rule_number,
        width=args.width,
        boundary=args.boundary,
        recurrence=args.recurrence,
        itr=args.itr,
        d_period=args.d_period,
        repeats=args.repeats,
        seed=args.seed,
        feature_mode=args.feature_mode,
        output_window=args.output_window,
    )
    reservoir_models = train_reservoir_linear_models(X_res, Y_res)
    reservoir_res = evaluate_models(reservoir_models, X_res, Y_res)
    _print_results("Reservoir", reservoir_res)

    if reservoir_res["full_correct"] != reservoir_res["n_samples"]:
        incorrect = reservoir_res["n_samples"] - reservoir_res["full_correct"]
        print(f"Reservoir full-vector incorrect: {incorrect}")


if __name__ == "__main__":
    main()
