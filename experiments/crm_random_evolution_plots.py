"""Random-CRM evolution smoke test with plots.

Run:
    python -m experiments.crm_random_evolution_plots
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
class CRMRunSummary:
    run_id: int
    seed: int
    n_species: int
    n_resources: int
    height: int
    width_grid: int
    proj_width: int
    mean_fitness_curve: list[float]
    best_fitness_curve: list[float]
    final_mean_fitness: float
    final_best_fitness: float


def _random_crm_config(rng: np.random.Generator) -> dict:
    n_species = int(rng.integers(2, 5))
    n_resources = int(rng.integers(2, 5))
    height = int(rng.integers(6, 11))
    width_grid = int(rng.integers(6, 11))
    proj_width = int(rng.integers(64, 193))

    reaction = rng.uniform(0.05, 0.45, size=(n_species, n_resources)).astype(np.float32)
    consumption = rng.uniform(0.05, 0.30, size=(n_species, n_resources)).astype(np.float32)
    inflow = rng.uniform(0.05, 0.35, size=(n_resources,)).astype(np.float32)
    diff_species = rng.uniform(0.005, 0.05, size=(n_species,)).astype(np.float32)
    diff_resources = rng.uniform(0.01, 0.08, size=(n_resources,)).astype(np.float32)

    return {
        "height": height,
        "width_grid": width_grid,
        "n_species": n_species,
        "n_resources": n_resources,
        "reaction_matrix": reaction.tolist(),
        "consumption_matrix": consumption.tolist(),
        "resource_inflow": inflow.tolist(),
        "diffusion_species": diff_species.tolist(),
        "diffusion_resources": diff_resources.tolist(),
        "dt": float(rng.uniform(0.02, 0.08)),
        "dilution": float(rng.uniform(0.002, 0.03)),
        "noise_std": float(rng.uniform(0.0, 0.01)),
        "inject_scale": float(rng.uniform(0.02, 0.08)),
        "projection": {
            "kind": "random",
            "output_width": proj_width,
            "seed": int(rng.integers(0, 2**31 - 1)),
            "scale": 1.0,
        },
    }


def _run_single(
    *,
    run_id: int,
    seed: int,
    bits: int,
    generations: int,
    population_size: int,
    support_size: int,
    challenge_size: int,
) -> tuple[CRMRunSummary, dict]:
    rng = np.random.default_rng(seed)
    cfg = _random_crm_config(rng)
    height = int(cfg["height"])
    width_grid = int(cfg["width_grid"])

    result = run_microbiome_host_evolution(
        task="bit_memory",
        source="reservoir",
        dataset_kwargs={
            "bits": bits,
            "rule_number": 110,
            "width": height * width_grid,
            "boundary": "periodic",
            "recurrence": 4,
            "itr": 2,
            "d_period": 20,
        },
        learner_kind="svm",
        learner_config=None,
        population_size=population_size,
        generations=generations,
        support_size=support_size,
        challenge_size=challenge_size,
        fitness_metric="full_vector_accuracy",
        shared_challenge_across_population=True,
        mutate_rule_prob=0.0,
        mutate_seed_prob=0.2,
        seed=seed,
        progress=False,
        inner_progress=False,
        use_dataset_cache=True,
        reservoir_kind="crm",
        reservoir_config=cfg,
    )

    mean_curve = [float(h.mean_fitness) for h in result.history]
    best_curve = [float(h.best_fitness) for h in result.history]
    summary = CRMRunSummary(
        run_id=run_id,
        seed=seed,
        n_species=int(cfg["n_species"]),
        n_resources=int(cfg["n_resources"]),
        height=height,
        width_grid=width_grid,
        proj_width=int(cfg["projection"]["output_width"]),
        mean_fitness_curve=mean_curve,
        best_fitness_curve=best_curve,
        final_mean_fitness=float(mean_curve[-1]),
        final_best_fitness=float(best_curve[-1]),
    )
    return summary, cfg


def _plot_curves(summaries: list[CRMRunSummary], out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for s in summaries:
        x = np.arange(len(s.best_fitness_curve))
        plt.plot(x, s.best_fitness_curve, alpha=0.85, label=f"run{s.run_id} best")
        plt.plot(x, s.mean_fitness_curve, alpha=0.35, linestyle="--", color=plt.gca().lines[-1].get_color())
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("CRM random configs: best (solid) and mean (dashed)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    if len(summaries) <= 10:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "crm_random_evolution_curves.png", dpi=180)
    plt.close()


def _plot_final_scores(summaries: list[CRMRunSummary], out_dir: Path) -> None:
    x = np.arange(len(summaries))
    best = np.array([s.final_best_fitness for s in summaries], dtype=float)
    mean = np.array([s.final_mean_fitness for s in summaries], dtype=float)

    plt.figure(figsize=(9, 4))
    width = 0.38
    plt.bar(x - width / 2, best, width=width, label="final best")
    plt.bar(x + width / 2, mean, width=width, label="final mean")
    labels = [f"r{s.run_id}\nS{s.n_species}/R{s.n_resources}" for s in summaries]
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("fitness")
    plt.title("Final fitness per random CRM configuration")
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "crm_random_evolution_final_scores.png", dpi=180)
    plt.close()


def _plot_structure_scatter(summaries: list[CRMRunSummary], out_dir: Path) -> None:
    species = np.array([s.n_species for s in summaries], dtype=float)
    resources = np.array([s.n_resources for s in summaries], dtype=float)
    proj = np.array([s.proj_width for s in summaries], dtype=float)
    final_best = np.array([s.final_best_fitness for s in summaries], dtype=float)

    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(species, resources, c=final_best, s=proj / 2.5, cmap="viridis", alpha=0.9)
    for s in summaries:
        plt.text(s.n_species + 0.03, s.n_resources + 0.03, f"r{s.run_id}", fontsize=8)
    plt.xlabel("n_species")
    plt.ylabel("n_resources")
    plt.title("CRM structure vs final best fitness (size = projection width)")
    plt.grid(True, alpha=0.25)
    cb = plt.colorbar(sc)
    cb.set_label("final best fitness")
    plt.tight_layout()
    plt.savefig(out_dir / "crm_random_evolution_structure_scatter.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Random CRM evolution smoke + plots")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of random CRM configs")
    parser.add_argument("--bits", type=int, default=5, help="Bit-memory size")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--support_size", type=int, default=64)
    parser.add_argument("--challenge_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/crm_random_evolution",
        help="Directory for plots and summary JSON",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    run_seeds = master_rng.integers(0, 2**31 - 1, size=args.n_runs, dtype=np.int64).tolist()

    summaries: list[CRMRunSummary] = []
    configs: list[dict] = []
    for run_id, run_seed in enumerate(run_seeds):
        summary, cfg = _run_single(
            run_id=run_id,
            seed=int(run_seed),
            bits=args.bits,
            generations=args.generations,
            population_size=args.population_size,
            support_size=args.support_size,
            challenge_size=args.challenge_size,
        )
        summaries.append(summary)
        configs.append(cfg)
        print(
            f"run {run_id}: final_best={summary.final_best_fitness:.4f} "
            f"(S={summary.n_species}, R={summary.n_resources}, proj={summary.proj_width})"
        )

    _plot_curves(summaries, out_dir)
    _plot_final_scores(summaries, out_dir)
    _plot_structure_scatter(summaries, out_dir)

    payload = {
        "config": {
            "n_runs": args.n_runs,
            "bits": args.bits,
            "generations": args.generations,
            "population_size": args.population_size,
            "support_size": args.support_size,
            "challenge_size": args.challenge_size,
            "seed": args.seed,
        },
        "summaries": [asdict(s) for s in summaries],
        "crm_configs": configs,
    }
    summary_path = out_dir / "crm_random_evolution_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved summary: {summary_path}")
    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
