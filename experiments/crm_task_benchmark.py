"""CRM-only full-dataset benchmark and optimization.

This benchmark answers: can CRM reservoir + readout fully learn a task when
training uses all available task combinations.

Run:
    python -m experiments.crm_task_benchmark --task bit_memory
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

from computingMicrobiome.feature_sources import (
    build_reservoir_bit_memory_dataset,
    build_reservoir_compound_opcode_dataset,
    build_reservoir_opcode_logic16_dataset,
    build_reservoir_opcode_logic_dataset,
    build_reservoir_serial_adder_dataset,
)
from computingMicrobiome.readouts.factory import make_readout

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DatasetBuilder = Callable[..., tuple[np.ndarray, np.ndarray, dict]]


TASK_BUILDERS: dict[str, DatasetBuilder] = {
    "bit_memory": build_reservoir_bit_memory_dataset,
    "opcode_logic": build_reservoir_opcode_logic_dataset,
    "opcode_logic16": build_reservoir_opcode_logic16_dataset,
    "compound_opcode": build_reservoir_compound_opcode_dataset,
    "serial_adder": build_reservoir_serial_adder_dataset,
}


def _default_dataset_kwargs(task: str) -> dict:
    if task == "bit_memory":
        return {
            "bits": 8,
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
        }
    if task == "opcode_logic":
        return {
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
            "repeats": 3,
            "feature_mode": "cue_tick",
            "output_window": 1,
        }
    if task == "opcode_logic16":
        return {
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
            "repeats": 5,
            "feature_mode": "cue_tick",
            "output_window": 1,
        }
    if task == "compound_opcode":
        return {
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 20,
            "repeats": 3,
            "feature_mode": "cue_tick",
            "output_window": 1,
        }
    if task == "serial_adder":
        return {
            "bits": 8,
            "n_samples": 256,
            "rule_number": 110,
            "width": 256,
            "boundary": "periodic",
            "recurrence": 8,
            "itr": 8,
            "d_period": 3,
        }
    raise ValueError(f"Unsupported task: {task}")


def _pick_grid(width_target: int, rng: np.random.Generator) -> tuple[int, int]:
    width_target = max(36, int(width_target))
    side = int(np.sqrt(width_target))
    candidates: list[tuple[int, int]] = []
    for h in range(max(6, side - 4), side + 5):
        w = max(6, int(round(width_target / h)))
        candidates.append((h, w))
    return candidates[int(rng.integers(0, len(candidates)))]


def _sample_crm_config(
    *, width_target: int, rng: np.random.Generator, trial_id: int
) -> tuple[dict, int]:
    n_species = int(rng.integers(2, 7))
    n_resources = int(rng.integers(2, 6))
    height, width_grid = _pick_grid(width_target, rng)
    width_cells = int(height * width_grid)

    reaction = rng.uniform(0.03, 0.50, size=(n_species, n_resources)).astype(np.float32)
    consumption = rng.uniform(0.03, 0.40, size=(n_species, n_resources)).astype(np.float32)
    inflow = rng.uniform(0.03, 0.40, size=(n_resources,)).astype(np.float32)
    diff_species = rng.uniform(0.002, 0.08, size=(n_species,)).astype(np.float32)
    diff_resources = rng.uniform(0.005, 0.10, size=(n_resources,)).astype(np.float32)

    projection_width = int(rng.choice([128, 192, 256, 320, 384]))
    cfg = {
        "height": height,
        "width_grid": width_grid,
        "n_species": n_species,
        "n_resources": n_resources,
        "reaction_matrix": reaction.tolist(),
        "consumption_matrix": consumption.tolist(),
        "resource_inflow": inflow.tolist(),
        "diffusion_species": diff_species.tolist(),
        "diffusion_resources": diff_resources.tolist(),
        "dt": float(rng.uniform(0.015, 0.08)),
        "dilution": float(rng.uniform(0.001, 0.04)),
        "noise_std": float(rng.uniform(0.0, 0.015)),
        "inject_scale": float(rng.uniform(0.01, 0.08)),
        "projection": {
            "kind": "random",
            "output_width": projection_width,
            "seed": int(10_000 + trial_id),
            "scale": 1.0,
        },
    }
    return cfg, width_cells


def _sample_dataset_kwargs(base_kwargs: dict, *, width_cells: int, rng: np.random.Generator) -> dict:
    kwargs = dict(base_kwargs)
    kwargs["width"] = int(width_cells)
    if "itr" in kwargs:
        kwargs["itr"] = int(rng.choice([4, 6, 8, 10, 12]))
    if "recurrence" in kwargs:
        kwargs["recurrence"] = int(rng.choice([4, 6, 8, 10]))
    if "d_period" in kwargs:
        kwargs["d_period"] = int(rng.choice([8, 12, 16, 20, 24]))
    if "repeats" in kwargs:
        kwargs["repeats"] = int(rng.choice([1, 2, 3, 5]))
    return kwargs


def _sample_readout_config(rng: np.random.Generator) -> dict:
    return {"C": float(10 ** rng.uniform(-2.0, 1.7)), "class_weight": "balanced"}


def _fit_and_score_full(
    X: np.ndarray,
    y: np.ndarray,
    *,
    readout_kind: str,
    readout_config: dict,
    fit_seed: int,
) -> float:
    rng = np.random.default_rng(fit_seed)
    reg = make_readout(readout_kind, readout_config, rng=rng)
    reg.fit(X, y)
    return float(reg.score(X, y))


@dataclass
class TrialSummary:
    trial_id: int
    mean_full_acc: float
    std_full_acc: float
    crm_height: int
    crm_width_grid: int
    crm_n_species: int
    crm_n_resources: int
    crm_proj_width: int
    itr: int
    recurrence: int
    d_period: int
    repeats: int
    readout_C: float
    dataset_seeds: list[int]
    fit_seeds: list[int]
    per_run_full_acc: list[float]
    reservoir_config: dict
    dataset_kwargs: dict
    readout_config: dict


def run_random_search(
    *,
    task: str,
    n_trials: int,
    eval_repeats: int,
    seed: int,
    readout_kind: str,
    progress: bool,
    progress_every: int,
) -> tuple[list[TrialSummary], dict]:
    if task not in TASK_BUILDERS:
        raise ValueError(f"Unsupported task: {task}. Valid: {sorted(TASK_BUILDERS)}")
    builder = TASK_BUILDERS[task]
    base_kwargs = _default_dataset_kwargs(task)
    width_target = int(base_kwargs.get("width", 256))

    master = np.random.default_rng(seed)
    trial_summaries: list[TrialSummary] = []
    run_records: list[dict] = []
    total_jobs = int(n_trials) * int(eval_repeats)
    jobs_done = 0
    t0 = time.perf_counter()

    if progress:
        print(
            f"[crm-benchmark] task={task} trials={n_trials} repeats={eval_repeats} "
            f"total_jobs={total_jobs}"
        )

    for trial_id in range(n_trials):
        trial_t0 = time.perf_counter()
        trng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        crm_cfg, width_cells = _sample_crm_config(width_target=width_target, rng=trng, trial_id=trial_id)
        dataset_kwargs = _sample_dataset_kwargs(base_kwargs, width_cells=width_cells, rng=trng)
        readout_cfg = _sample_readout_config(trng)

        full_accs: list[float] = []
        dataset_seeds: list[int] = []
        fit_seeds: list[int] = []

        for rep_idx in range(eval_repeats):
            dataset_seed = int(master.integers(0, 2**31 - 1))
            fit_seed = int(master.integers(0, 2**31 - 1))
            dataset_seeds.append(dataset_seed)
            fit_seeds.append(fit_seed)

            kwargs = dict(dataset_kwargs)
            kwargs["seed"] = dataset_seed
            X, y, _meta = builder(reservoir_kind="crm", reservoir_config=crm_cfg, **kwargs)
            X = np.asarray(X)
            y = np.asarray(y).reshape(-1)

            full_acc = _fit_and_score_full(
                X,
                y,
                readout_kind=readout_kind,
                readout_config=readout_cfg,
                fit_seed=fit_seed,
            )
            full_accs.append(full_acc)
            run_records.append(
                {
                    "trial_id": trial_id,
                    "repeat_idx": rep_idx,
                    "dataset_seed": dataset_seed,
                    "fit_seed": fit_seed,
                    "full_acc": full_acc,
                    "n_rows": int(X.shape[0]),
                }
            )
            jobs_done += 1

            if progress and (jobs_done % max(1, progress_every) == 0 or jobs_done == total_jobs):
                elapsed = time.perf_counter() - t0
                rate = elapsed / max(1, jobs_done)
                eta = rate * (total_jobs - jobs_done)
                print(
                    f"[crm-benchmark] {jobs_done}/{total_jobs} jobs "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"last_full_acc={full_acc:.4f}"
                )

        trial_summaries.append(
            TrialSummary(
                trial_id=trial_id,
                mean_full_acc=float(np.mean(full_accs)),
                std_full_acc=float(np.std(full_accs)),
                crm_height=int(crm_cfg["height"]),
                crm_width_grid=int(crm_cfg["width_grid"]),
                crm_n_species=int(crm_cfg["n_species"]),
                crm_n_resources=int(crm_cfg["n_resources"]),
                crm_proj_width=int(crm_cfg["projection"]["output_width"]),
                itr=int(dataset_kwargs.get("itr", -1)),
                recurrence=int(dataset_kwargs.get("recurrence", -1)),
                d_period=int(dataset_kwargs.get("d_period", -1)),
                repeats=int(dataset_kwargs.get("repeats", 1)),
                readout_C=float(readout_cfg["C"]),
                dataset_seeds=dataset_seeds,
                fit_seeds=fit_seeds,
                per_run_full_acc=full_accs,
                reservoir_config=crm_cfg,
                dataset_kwargs=dataset_kwargs,
                readout_config=readout_cfg,
            )
        )

        if progress:
            trial_elapsed = time.perf_counter() - trial_t0
            trial_mean = float(np.mean(full_accs))
            trial_std = float(np.std(full_accs))
            print(
                f"[crm-benchmark] trial {trial_id + 1}/{n_trials} done "
                f"mean_full_acc={trial_mean:.4f} std={trial_std:.4f} "
                f"trial_time={trial_elapsed:.1f}s"
            )

    trial_summaries.sort(key=lambda x: x.mean_full_acc, reverse=True)
    return trial_summaries, {"per_run": run_records}


def _plot_search_results(summaries: list[TrialSummary], out_dir: Path, *, top_k: int = 12) -> None:
    scores = np.array([s.mean_full_acc for s in summaries], dtype=float)
    stds = np.array([s.std_full_acc for s in summaries], dtype=float)
    ids = np.array([s.trial_id for s in summaries], dtype=int)
    species = np.array([s.crm_n_species for s in summaries], dtype=float)
    resources = np.array([s.crm_n_resources for s in summaries], dtype=float)
    proj = np.array([s.crm_proj_width for s in summaries], dtype=float)

    plt.figure(figsize=(9, 4.5))
    order = np.arange(scores.size)
    plt.errorbar(order, scores, yerr=stds, fmt="o", alpha=0.8, capsize=2)
    plt.xlabel("ranked trial index")
    plt.ylabel("mean full-dataset accuracy")
    plt.title("CRM search results (error bars = std over repeats)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "crm_search_ranked_scores.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(species, resources, c=scores, s=proj / 2.0, cmap="viridis", alpha=0.9)
    for i in range(min(len(ids), 20)):
        plt.text(species[i] + 0.03, resources[i] + 0.03, f"t{ids[i]}", fontsize=8)
    plt.xlabel("n_species")
    plt.ylabel("n_resources")
    plt.title("CRM structure vs mean full accuracy")
    plt.grid(True, alpha=0.25)
    cb = plt.colorbar(sc)
    cb.set_label("mean full accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "crm_search_structure_scatter.png", dpi=180)
    plt.close()

    k = min(top_k, len(summaries))
    top = summaries[:k]
    x = np.arange(k)
    vals = [t.mean_full_acc for t in top]
    err = [t.std_full_acc for t in top]
    labels = [f"t{t.trial_id}" for t in top]
    plt.figure(figsize=(10, 4.5))
    plt.bar(x, vals, yerr=err, capsize=3)
    plt.xticks(x, labels)
    plt.ylabel("mean full-dataset accuracy")
    plt.title(f"Top-{k} CRM trials")
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "crm_search_top_trials.png", dpi=180)
    plt.close()


def _write_outputs(
    *,
    out_dir: Path,
    task: str,
    args: argparse.Namespace,
    summaries: list[TrialSummary],
    extras: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "config": {
            "n_trials": args.n_trials,
            "eval_repeats": args.eval_repeats,
            "seed": args.seed,
            "readout_kind": args.readout_kind,
            "evaluation_mode": "full_dataset_fit_and_score",
            "note_on_trials": (
                "n_trials controls random-search budget; 24 is a default compromise "
                "between coverage and runtime, not a required value."
            ),
        },
        "top_trial": asdict(summaries[0]),
        "summaries": [asdict(s) for s in summaries],
        "extras": extras,
    }
    (out_dir / "crm_task_benchmark_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    csv_path = out_dir / "crm_task_benchmark_trials.csv"
    csv_fields = [
        "trial_id",
        "mean_full_acc",
        "std_full_acc",
        "crm_height",
        "crm_width_grid",
        "crm_n_species",
        "crm_n_resources",
        "crm_proj_width",
        "itr",
        "recurrence",
        "d_period",
        "repeats",
        "readout_C",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: getattr(s, k) for k in csv_fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="CRM-only full-dataset benchmark and optimization")
    parser.add_argument(
        "--task",
        type=str,
        default="bit_memory",
        choices=sorted(TASK_BUILDERS.keys()),
        help="Task to benchmark using CRM reservoir only",
    )
    parser.add_argument("--n_trials", type=int, default=24)
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--readout_kind", type=str, default="svm")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress, elapsed time, and ETA",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="Print progress every N jobs (job = one trial repeat)",
    )
    parser.add_argument("--out_dir", type=str, default="artifacts/crm_task_benchmark")
    args = parser.parse_args()

    summaries, extras = run_random_search(
        task=args.task,
        n_trials=args.n_trials,
        eval_repeats=args.eval_repeats,
        seed=args.seed,
        readout_kind=args.readout_kind,
        progress=bool(args.progress),
        progress_every=max(1, int(args.progress_every)),
    )
    out_dir = Path(args.out_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_search_results(summaries, out_dir)
    _write_outputs(
        out_dir=out_dir,
        task=args.task,
        args=args,
        summaries=summaries,
        extras=extras,
    )

    top = summaries[0]
    print(
        "Best trial:",
        top.trial_id,
        "mean_full_acc=",
        round(top.mean_full_acc, 4),
        "std=",
        round(top.std_full_acc, 4),
    )
    print("Saved artifacts in:", out_dir)


if __name__ == "__main__":
    main()
