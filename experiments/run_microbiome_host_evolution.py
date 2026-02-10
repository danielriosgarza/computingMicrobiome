"""Run microbiome-centric host evolution experiments.

Example:
    python -m experiments.run_microbiome_host_evolution --task bit_memory --source reservoir
    python -m experiments.run_microbiome_host_evolution --task bit_memory --source direct
"""

from __future__ import annotations

import argparse
from pathlib import Path

from computingMicrobiome.evolution import run_microbiome_host_evolution


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbiome host evolution")
    parser.add_argument(
        "--task",
        type=str,
        default="bit_memory",
        choices=[
            "bit_memory",
            "opcode_logic",
            "opcode_logic16",
            "compound_opcode",
            "serial_adder",
            "toy_addition",
        ],
    )
    parser.add_argument("--source", type=str, default="reservoir", choices=["direct", "reservoir"])
    parser.add_argument("--population_size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--support_size", type=int, default=128)
    parser.add_argument("--challenge_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learner_kind", type=str, default="svm")
    parser.add_argument(
        "--fitness_metric",
        type=str,
        default="full_vector_accuracy",
        choices=["full_vector_accuracy", "mean_bit_accuracy"],
    )
    parser.add_argument("--mutate_rule_prob", type=float, default=0.0)
    parser.add_argument("--mutate_seed_prob", type=float, default=0.2)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--progress_mode",
        type=str,
        default="auto",
        choices=["auto", "tqdm", "print"],
    )
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--no_inner_progress", action="store_true")
    parser.add_argument("--inner_progress_every", type=int, default=5)
    parser.add_argument("--no_dataset_cache", action="store_true")
    parser.add_argument("--out", type=str, default="artifacts/microbiome_evolution_result.json")
    args = parser.parse_args()

    # Conservative task defaults so the script runs out-of-the-box.
    dataset_kwargs = {
        "bit_memory": {
            "bits": 8,
            "rule_number": 110,
            "width": 700,
            "boundary": "periodic",
            "recurrence": 4,
            "itr": 2,
            "d_period": 200,
        },
        "opcode_logic": {
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
            "repeats": 1,
            "feature_mode": "cue_tick",
            "output_window": 2,
        },
        "opcode_logic16": {
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
            "repeats": 1,
            "feature_mode": "cue_tick",
            "output_window": 2,
        },
        "compound_opcode": {
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
            "repeats": 1,
            "feature_mode": "cue_tick",
            "output_window": 2,
        },
        "serial_adder": {
            "bits": 8,
            "n_samples": 512,
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
        },
        "toy_addition": {
            "n_bits": 3,
            "cin": 0,
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
            "repeats": 1,
            "feature_mode": "cue_tick",
            "output_window": 2,
        },
    }

    result = run_microbiome_host_evolution(
        task=args.task,
        source=args.source,
        dataset_kwargs=dataset_kwargs[args.task],
        learner_kind=args.learner_kind,
        learner_config=None,
        population_size=args.population_size,
        generations=args.generations,
        support_size=args.support_size,
        challenge_size=args.challenge_size,
        fitness_metric=args.fitness_metric,
        shared_challenge_across_population=True,
        mutate_rule_prob=args.mutate_rule_prob,
        mutate_seed_prob=args.mutate_seed_prob,
        seed=args.seed,
        progress=args.progress,
        progress_mode=args.progress_mode,
        progress_every=args.progress_every,
        inner_progress=not args.no_inner_progress,
        inner_progress_every=args.inner_progress_every,
        use_dataset_cache=not args.no_dataset_cache,
    )

    out_path = Path(args.out)
    result.save_json(out_path)

    print("Saved:", out_path)
    print("Best fitness:", max(m.best_fitness for m in result.history))
    print("Final mean fitness:", result.history[-1].mean_fitness)
    print("Best genotype:", result.best_genotype)


if __name__ == "__main__":
    main()
