"""CRM-only escalation search to reach perfect score.

Goal:
  Find the first CRM configuration that reaches perfect fitness (1.0) on a
  target task, with optional multi-seed consistency checks.

Run:
  python -m experiments.crm_only_to_perfect
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

from computingMicrobiome.evolution import run_microbiome_host_evolution

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class AttemptResult:
    stage_idx: int
    seed: int
    best_fitness: float
    final_mean_fitness: float
    generations: int
    config_id: str


def _make_crm_config(
    *,
    height: int,
    width_grid: int,
    n_species: int,
    n_resources: int,
    projection_width: int,
    seed: int,
    dt: float = 0.05,
    dilution: float = 0.01,
    noise_std: float = 0.002,
) -> dict:
    rng = np.random.default_rng(seed)
    reaction = rng.uniform(0.05, 0.45, size=(n_species, n_resources)).astype(np.float32)
    consumption = rng.uniform(0.05, 0.30, size=(n_species, n_resources)).astype(np.float32)
    inflow = rng.uniform(0.08, 0.30, size=(n_resources,)).astype(np.float32)
    diff_species = rng.uniform(0.01, 0.05, size=(n_species,)).astype(np.float32)
    diff_resources = rng.uniform(0.02, 0.08, size=(n_resources,)).astype(np.float32)

    return {
        "height": int(height),
        "width_grid": int(width_grid),
        "n_species": int(n_species),
        "n_resources": int(n_resources),
        "reaction_matrix": reaction.tolist(),
        "consumption_matrix": consumption.tolist(),
        "resource_inflow": inflow.tolist(),
        "diffusion_species": diff_species.tolist(),
        "diffusion_resources": diff_resources.tolist(),
        "dt": float(dt),
        "dilution": float(dilution),
        "noise_std": float(noise_std),
        "inject_scale": 0.05,
        "projection": {
            "kind": "random",
            "output_width": int(projection_width),
            "seed": int(seed),
            "scale": 1.0,
        },
    }


def _staged_configs(base_seed: int) -> list[tuple[str, dict]]:
    # Escalation order: increase representational capacity gradually.
    stages = [
        ("s1_small", dict(height=16, width_grid=16, n_species=4, n_resources=3, projection_width=256)),
        ("s2_proj_up", dict(height=16, width_grid=16, n_species=4, n_resources=3, projection_width=384)),
        ("s3_species_up", dict(height=16, width_grid=16, n_species=6, n_resources=4, projection_width=384)),
        ("s4_proj_512", dict(height=16, width_grid=16, n_species=6, n_resources=4, projection_width=512)),
        ("s5_grid_up", dict(height=18, width_grid=18, n_species=6, n_resources=5, projection_width=512)),
    ]
    out: list[tuple[str, dict]] = []
    for idx, (name, s) in enumerate(stages):
        cfg = _make_crm_config(
            **s,
            seed=base_seed + 1000 + idx,
            dt=0.05,
            dilution=0.01,
            noise_std=0.0015,
        )
        out.append((name, cfg))
    return out


def _run_one(
    *,
    task: str,
    bits: int,
    generations: int,
    population_size: int,
    support_size: int,
    challenge_size: int,
    seed: int,
    crm_config: dict,
):
    width = int(crm_config["height"] * crm_config["width_grid"])
    dataset_kwargs = {
        "bits": bits,
        "rule_number": 110,  # retained for builder compatibility
        "width": width,
        "boundary": "periodic",
        "recurrence": 8,
        "itr": 8,
        "d_period": 20,
    }
    return run_microbiome_host_evolution(
        task=task,
        source="reservoir",
        dataset_kwargs=dataset_kwargs,
        learner_kind="svm",
        learner_config=None,
        population_size=population_size,
        generations=generations,
        support_size=support_size,
        challenge_size=challenge_size,
        fitness_metric="full_vector_accuracy",
        shared_challenge_across_population=True,
        mutate_rule_prob=0.0,
        mutate_seed_prob=0.0,
        seed=seed,
        progress=False,
        inner_progress=False,
        use_dataset_cache=True,
        reservoir_kind="crm",
        reservoir_config=crm_config,
    )


def _plot_best_curves(curves: list[tuple[str, list[float]]], out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for label, y in curves:
        plt.plot(np.arange(len(y)), y, label=label)
    plt.xlabel("generation")
    plt.ylabel("best fitness")
    plt.title("CRM-only escalation: best fitness curves")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "crm_only_to_perfect_curves.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CRM-only search for perfect score")
    parser.add_argument("--task", type=str, default="bit_memory", choices=["bit_memory"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population_size", type=int, default=80)
    parser.add_argument("--support_size", type=int, default=128)
    parser.add_argument("--challenge_size", type=int, default=128)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[7, 11, 23],
        help="Seeds for consistency check at each stage",
    )
    parser.add_argument(
        "--required_successes",
        type=int,
        default=2,
        help="How many seeds must hit 1.0 to stop early",
    )
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--out_dir", type=str, default="artifacts/crm_only_to_perfect")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_cfgs = _staged_configs(args.base_seed)
    attempts: list[AttemptResult] = []
    curves: list[tuple[str, list[float]]] = []
    winner = None

    print(
        f"[crm-only] task={args.task} bits={args.bits} stages={len(stage_cfgs)} "
        f"seeds={args.seeds} required_successes={args.required_successes}"
    )

    for stage_idx, (stage_name, cfg) in enumerate(stage_cfgs):
        success_count = 0
        print(f"[crm-only] stage {stage_idx + 1}/{len(stage_cfgs)}: {stage_name}")
        for seed in args.seeds:
            res = _run_one(
                task=args.task,
                bits=args.bits,
                generations=args.generations,
                population_size=args.population_size,
                support_size=args.support_size,
                challenge_size=args.challenge_size,
                seed=int(seed),
                crm_config=cfg,
            )
            best_curve = [float(h.best_fitness) for h in res.history]
            best = float(max(best_curve))
            final_mean = float(res.history[-1].mean_fitness)
            attempts.append(
                AttemptResult(
                    stage_idx=stage_idx,
                    seed=int(seed),
                    best_fitness=best,
                    final_mean_fitness=final_mean,
                    generations=int(args.generations),
                    config_id=stage_name,
                )
            )
            curves.append((f"{stage_name}_seed{seed}", best_curve))
            if best >= 1.0:
                success_count += 1
            print(
                f"[crm-only]   seed={seed} best={best:.4f} final_mean={final_mean:.4f} "
                f"successes={success_count}/{args.required_successes}"
            )
            if success_count >= int(args.required_successes):
                winner = {"stage_name": stage_name, "stage_idx": stage_idx, "crm_config": cfg}
                break
        if winner is not None:
            break

    _plot_best_curves(curves, out_dir)

    payload = {
        "config": {
            "task": args.task,
            "bits": args.bits,
            "generations": args.generations,
            "population_size": args.population_size,
            "support_size": args.support_size,
            "challenge_size": args.challenge_size,
            "seeds": args.seeds,
            "required_successes": args.required_successes,
            "base_seed": args.base_seed,
        },
        "winner": winner,
        "attempts": [asdict(a) for a in attempts],
        "stage_configs": [{"name": n, "config": c} for n, c in stage_cfgs],
    }
    out_path = out_dir / "crm_only_to_perfect_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if winner is None:
        print("[crm-only] no perfect config reached required seed consistency")
    else:
        print(
            f"[crm-only] winner={winner['stage_name']} at stage {winner['stage_idx'] + 1} "
            f"(>= {args.required_successes} perfect seeds)"
        )
    print("Saved:", out_path)
    print("Saved curves:", out_dir / "crm_only_to_perfect_curves.png")


if __name__ == "__main__":
    main()
