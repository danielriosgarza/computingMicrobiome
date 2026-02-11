"""Sanity checks for rule-90 vs rule-110 behavior on 8-bit memory.

This script compares:
1) Plain SVM (no population loop) at fixed seed.
2) Plain SVM seed sweep.
3) Microbiome host evolution in a one-generation, no-mutation regime.

Run:
    python -m experiments.rule90_rule110_sanity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from computingMicrobiome.evolution import run_microbiome_host_evolution
from computingMicrobiome.feature_sources import build_reservoir_bit_memory_dataset
from computingMicrobiome.readouts.factory import make_readout


def _full_dataset_accuracy(
    *,
    rule_number: int,
    bits: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    dataset_seed: int,
    learner_kind: str,
    learner_config: dict | None,
) -> dict:
    X, y, _meta = build_reservoir_bit_memory_dataset(
        bits=bits,
        rule_number=rule_number,
        width=width,
        boundary=boundary,
        recurrence=recurrence,
        itr=itr,
        d_period=d_period,
        seed=dataset_seed,
    )
    reg = make_readout(learner_kind, learner_config, rng=np.random.default_rng(dataset_seed))
    reg.fit(X, y)
    acc = float(reg.score(X, y))
    return {"rule": rule_number, "seed": dataset_seed, "n_rows": int(X.shape[0]), "acc": acc}


def _sweep_full_dataset_accuracy(
    *,
    rule_number: int,
    seeds: list[int],
    bits: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    learner_kind: str,
    learner_config: dict | None,
) -> dict:
    rows = []
    for s in seeds:
        rows.append(
            _full_dataset_accuracy(
                rule_number=rule_number,
                bits=bits,
                width=width,
                boundary=boundary,
                recurrence=recurrence,
                itr=itr,
                d_period=d_period,
                dataset_seed=s,
                learner_kind=learner_kind,
                learner_config=learner_config,
            )
        )
    accs = [r["acc"] for r in rows]
    return {
        "rule": rule_number,
        "n_seeds": len(seeds),
        "mean_acc": float(mean(accs)),
        "std_acc": float(pstdev(accs)) if len(accs) > 1 else 0.0,
        "rows": rows,
    }


def _population_one_generation(
    *,
    rule_number: int,
    bits: int,
    width: int,
    boundary: str,
    recurrence: int,
    itr: int,
    d_period: int,
    learner_kind: str,
    learner_config: dict | None,
    population_size: int,
    outer_seed: int,
) -> dict:
    # Full dataset rows for bit_memory reservoir condition.
    # n_samples episodes = 2**bits, each contributes `bits` output rows.
    full_rows = (2**bits) * bits

    res = run_microbiome_host_evolution(
        task="bit_memory",
        source="reservoir",
        dataset_kwargs={
            "bits": bits,
            "rule_number": rule_number,
            "width": width,
            "boundary": boundary,
            "recurrence": recurrence,
            "itr": itr,
            "d_period": d_period,
        },
        learner_kind=learner_kind,
        learner_config=learner_config,
        population_size=population_size,
        generations=1,
        support_size=full_rows,
        challenge_size=full_rows,
        fitness_metric="full_vector_accuracy",
        shared_challenge_across_population=True,
        mutate_rule_prob=0.0,
        mutate_seed_prob=0.0,
        seed=outer_seed,
        progress=False,
        inner_progress=False,
        use_dataset_cache=True,
    )

    h = res.history[0]
    return {
        "rule": rule_number,
        "population_size": population_size,
        "outer_seed": outer_seed,
        "generation": 0,
        "mean_fitness": float(h.mean_fitness),
        "best_fitness": float(h.best_fitness),
        "std_fitness": float(h.std_fitness),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule 90 vs 110 sanity checks")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--boundary", type=str, default="periodic")
    parser.add_argument("--recurrence", type=int, default=4)
    parser.add_argument("--itr", type=int, default=2)
    parser.add_argument("--d_period", type=int, default=200)
    parser.add_argument("--learner_kind", type=str, default="svm")
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--sweep_seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--outer_seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="artifacts/rule90_rule110_sanity.json")
    args = parser.parse_args()

    learner_config = None

    fixed_90 = _full_dataset_accuracy(
        rule_number=90,
        bits=args.bits,
        width=args.width,
        boundary=args.boundary,
        recurrence=args.recurrence,
        itr=args.itr,
        d_period=args.d_period,
        dataset_seed=args.dataset_seed,
        learner_kind=args.learner_kind,
        learner_config=learner_config,
    )
    fixed_110 = _full_dataset_accuracy(
        rule_number=110,
        bits=args.bits,
        width=args.width,
        boundary=args.boundary,
        recurrence=args.recurrence,
        itr=args.itr,
        d_period=args.d_period,
        dataset_seed=args.dataset_seed,
        learner_kind=args.learner_kind,
        learner_config=learner_config,
    )

    sweep_90 = _sweep_full_dataset_accuracy(
        rule_number=90,
        seeds=args.sweep_seeds,
        bits=args.bits,
        width=args.width,
        boundary=args.boundary,
        recurrence=args.recurrence,
        itr=args.itr,
        d_period=args.d_period,
        learner_kind=args.learner_kind,
        learner_config=learner_config,
    )
    sweep_110 = _sweep_full_dataset_accuracy(
        rule_number=110,
        seeds=args.sweep_seeds,
        bits=args.bits,
        width=args.width,
        boundary=args.boundary,
        recurrence=args.recurrence,
        itr=args.itr,
        d_period=args.d_period,
        learner_kind=args.learner_kind,
        learner_config=learner_config,
    )

    pop_90 = _population_one_generation(
        rule_number=90,
        bits=args.bits,
        width=args.width,
        boundary=args.boundary,
        recurrence=args.recurrence,
        itr=args.itr,
        d_period=args.d_period,
        learner_kind=args.learner_kind,
        learner_config=learner_config,
        population_size=args.population_size,
        outer_seed=args.outer_seed,
    )
    pop_110 = _population_one_generation(
        rule_number=110,
        bits=args.bits,
        width=args.width,
        boundary=args.boundary,
        recurrence=args.recurrence,
        itr=args.itr,
        d_period=args.d_period,
        learner_kind=args.learner_kind,
        learner_config=learner_config,
        population_size=args.population_size,
        outer_seed=args.outer_seed,
    )

    result = {
        "config": {
            "bits": args.bits,
            "width": args.width,
            "boundary": args.boundary,
            "recurrence": args.recurrence,
            "itr": args.itr,
            "d_period": args.d_period,
            "learner_kind": args.learner_kind,
            "dataset_seed": args.dataset_seed,
            "sweep_seeds": args.sweep_seeds,
            "population_size": args.population_size,
            "outer_seed": args.outer_seed,
        },
        "fixed_seed_full_dataset": {
            "rule90": fixed_90,
            "rule110": fixed_110,
            "delta_110_minus_90": float(fixed_110["acc"] - fixed_90["acc"]),
        },
        "seed_sweep_full_dataset": {
            "rule90": sweep_90,
            "rule110": sweep_110,
            "delta_mean_110_minus_90": float(sweep_110["mean_acc"] - sweep_90["mean_acc"]),
        },
        "population_one_generation_no_mutation": {
            "rule90": pop_90,
            "rule110": pop_110,
            "delta_mean_110_minus_90": float(pop_110["mean_fitness"] - pop_90["mean_fitness"]),
            "delta_best_110_minus_90": float(pop_110["best_fitness"] - pop_90["best_fitness"]),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("")
    print("[fixed seed full dataset]")
    print("rule 90 :", round(fixed_90["acc"], 6))
    print("rule 110:", round(fixed_110["acc"], 6))
    print("delta 110-90:", round(result["fixed_seed_full_dataset"]["delta_110_minus_90"], 6))
    print("")
    print("[seed sweep full dataset]")
    print("rule 90 mean :", round(sweep_90["mean_acc"], 6), "std:", round(sweep_90["std_acc"], 6))
    print("rule 110 mean:", round(sweep_110["mean_acc"], 6), "std:", round(sweep_110["std_acc"], 6))
    print("delta mean 110-90:", round(result["seed_sweep_full_dataset"]["delta_mean_110_minus_90"], 6))
    print("")
    print("[population, 1 generation, no mutation]")
    print("rule 90 mean/best :", round(pop_90["mean_fitness"], 6), round(pop_90["best_fitness"], 6))
    print("rule 110 mean/best:", round(pop_110["mean_fitness"], 6), round(pop_110["best_fitness"], 6))
    print(
        "delta mean/best 110-90:",
        round(result["population_one_generation_no_mutation"]["delta_mean_110_minus_90"], 6),
        round(result["population_one_generation_no_mutation"]["delta_best_110_minus_90"], 6),
    )


if __name__ == "__main__":
    main()

