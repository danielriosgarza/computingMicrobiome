"""Compare SVM and evolutionary readouts with and without reservoirs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from computingMicrobiome.readouts.factory import make_readout
from computingMicrobiome.readouts.base import Readout, coerce_xy_binary
from computingMicrobiome.readouts.evolutionary_linear import EvolutionaryLinearReadout
from computingMicrobiome.feature_sources import (
    build_direct_bit_memory_dataset,
    build_direct_compound_opcode_dataset,
    build_direct_opcode_logic16_dataset,
    build_direct_opcode_logic_dataset,
    build_direct_serial_adder_dataset,
    build_direct_toy_addition_dataset,
    build_reservoir_bit_memory_dataset,
    build_reservoir_compound_opcode_dataset,
    build_reservoir_opcode_logic16_dataset,
    build_reservoir_opcode_logic_dataset,
    build_reservoir_serial_adder_dataset,
    build_reservoir_toy_addition_dataset,
)


DatasetBuilder = Callable[..., Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray | None]]]

EVO_PRESETS: Dict[str, dict] = {
    "bit_memory": {
        "population_size": 512,
        "generations": 1500,
        "tournament_size": 7,
        "elite_count": 10,
        "mutation_scale": 0.05,
        "mutation_schedule": "linear_decay",
        "min_mutation_scale": 0.01,
        "mutation_fraction": 0.5,
        "batch_size": None,
    },
    "compound_opcode": {
        "population_size": 1024,
        "generations": 3000,
        "tournament_size": 7,
        "elite_count": 10,
        "mutation_scale": 0.06,
        "mutation_schedule": "linear_decay",
        "min_mutation_scale": 0.01,
        "mutation_fraction": 0.4,
        "batch_size": None,
    },
    "serial_adder": {
        "population_size": 512,
        "generations": 2000,
        "tournament_size": 7,
        "elite_count": 10,
        "mutation_scale": 0.06,
        "mutation_schedule": "linear_decay",
        "min_mutation_scale": 0.01,
        "mutation_fraction": 0.4,
        "batch_size": None,
    },
}

DEFAULT_EVO_CONFIG: Dict[str, dict] = {
    "population_size": 128,
    "generations": 300,
    "tournament_size": 5,
    "elite_count": 4,
    "mutation_scale": 0.1,
    "mutation_schedule": "linear_decay",
    "min_mutation_scale": 0.02,
    "mutation_fraction": 0.5,
    "batch_size": None,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    reservoir_builder: DatasetBuilder | None
    reservoir_kwargs: Dict
    direct_builder: DatasetBuilder | None
    direct_kwargs: Dict


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    pos = logits >= 0
    neg = ~pos
    out = np.empty_like(logits, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_x = np.exp(logits[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _binary_cross_entropy_prob(prob: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(prob, eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(loss.mean())


def _infer_label_mode(y: np.ndarray) -> dict:
    y_arr = np.asarray(y)
    info = {
        "y_shape": tuple(int(v) for v in y_arr.shape),
        "y_dtype": str(y_arr.dtype),
        "y_mode": "unknown",
        "y_class_hist": None,
        "y_pos_rate": None,
    }
    if y_arr.ndim == 1:
        unique, counts = np.unique(y_arr, return_counts=True)
        info["y_class_hist"] = list(zip(unique.tolist(), counts.astype(int).tolist()))
        if unique.size <= 2:
            info["y_mode"] = "binary"
        else:
            info["y_mode"] = "multiclass"
    elif y_arr.ndim == 2:
        info["y_mode"] = "multilabel"
        info["y_pos_rate"] = y_arr.mean(axis=0).astype(float).tolist()
    else:
        info["y_mode"] = "unknown"
    return info


def _budget_info(
    reg: Readout,
    readout_kind: str,
    readout_config: dict | None,
    n_samples: int,
) -> dict:
    if readout_kind != "evo":
        return {
            "budget_population_size": None,
            "budget_generations": None,
            "budget_batch_size": None,
            "budget_total_evaluations": None,
        }

    if isinstance(reg, EvolutionaryLinearReadout):
        pop = reg.population_size
        gens = reg.generations
        batch = reg.batch_size
    else:
        cfg = readout_config or {}
        pop = cfg.get("population_size")
        gens = cfg.get("generations")
        batch = cfg.get("batch_size")

    if batch is None:
        batch = n_samples
    total = None
    if pop is not None and gens is not None and batch is not None:
        total = int(pop) * int(gens) * int(batch)

    return {
        "budget_population_size": pop,
        "budget_generations": gens,
        "budget_batch_size": batch,
        "budget_total_evaluations": total,
    }


def _zscore_normalize(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    mean = X_arr.mean(axis=0, keepdims=True)
    std = X_arr.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (X_arr - mean) / std


def _subsample_features(
    X: np.ndarray, *, n_features: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if n_features <= 0 or n_features >= X.shape[1]:
        return X, np.arange(X.shape[1], dtype=int)
    idx = rng.choice(X.shape[1], size=n_features, replace=False)
    return X[:, idx], np.sort(idx)


def _train_single_readout(
    X: np.ndarray,
    y: np.ndarray,
    *,
    readout_kind: str,
    readout_config: dict | None,
    rng: np.random.Generator,
) -> Readout:
    reg = make_readout(readout_kind, readout_config, rng=rng)
    reg.fit(X, y)
    return reg


def _train_multi_output(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    readout_kind: str,
    readout_config: dict | None,
    rng: np.random.Generator,
) -> List[Readout]:
    models: List[Readout] = []
    for i in range(Y.shape[1]):
        reg = make_readout(readout_kind, readout_config, rng=rng)
        reg.fit(X, Y[:, i])
        models.append(reg)
    return models


def _evaluate_readout(
    reg: Readout,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    y_pred = reg.predict(X).reshape(-1)
    _, y_bin = coerce_xy_binary(X, y)
    y_bin = y_bin.astype(np.int8)
    logits = reg.decision_function(X).reshape(-1)
    prob = _sigmoid(logits)
    return {
        "per_bit": [float((y_pred == y_bin).mean())],
        "full_acc": float((y_pred == y_bin).mean()),
        "cross_entropy": _binary_cross_entropy_prob(prob, y_bin.astype(float)),
        "n_samples": int(y_bin.shape[0]),
    }


def _evaluate_multi_output(models: List[Readout], X: np.ndarray, Y: np.ndarray) -> dict:
    preds = np.zeros_like(Y, dtype=np.int8)
    logits_all = []
    for i, reg in enumerate(models):
        preds[:, i] = reg.predict(X).astype(np.int8)
        logits_all.append(reg.decision_function(X).reshape(-1))
    per_bit = (preds == Y).mean(axis=0).tolist()
    full_acc = float((preds == Y).all(axis=1).mean())
    logits_stack = np.vstack(logits_all).T
    prob = _sigmoid(logits_stack)
    ce = _binary_cross_entropy_prob(prob, Y.astype(float))
    return {
        "per_bit": per_bit,
        "full_acc": full_acc,
        "cross_entropy": ce,
        "n_samples": int(Y.shape[0]),
    }


def run_comparison(
    task_specs: Iterable[TaskSpec],
    *,
    readout_kinds: Iterable[str] = ("svm", "evo"),
    feature_sources: Iterable[str] = ("reservoir", "direct"),
    readout_configs: Dict[str, dict] | None = None,
    readout_configs_by_task: Dict[str, Dict[str, dict]] | None = None,
    normalize_features: bool = True,
    feature_subsample: int | None = None,
    seed: int = 0,
) -> List[dict]:
    results: List[dict] = []
    readout_configs = readout_configs or {}
    if "evo" not in readout_configs:
        readout_configs["evo"] = dict(DEFAULT_EVO_CONFIG)
    readout_configs_by_task = readout_configs_by_task or {}
    rng = np.random.default_rng(seed)

    for task in task_specs:
        for source in feature_sources:
            if source == "reservoir":
                builder = task.reservoir_builder
                kwargs = task.reservoir_kwargs
            elif source == "direct":
                builder = task.direct_builder
                kwargs = task.direct_kwargs
            else:
                raise ValueError(f"Unknown feature source: {source}")

            if builder is None:
                continue

            X, y, _meta = builder(**kwargs)
            label_info = _infer_label_mode(y)
            if label_info["y_mode"] == "multiclass":
                raise ValueError(
                    f"Multiclass labels are not supported yet for task={task.name}"
                )
            if normalize_features:
                X = _zscore_normalize(X)
            feature_idx = None
            if feature_subsample is not None:
                X, feature_idx = _subsample_features(
                    X, n_features=feature_subsample, rng=rng
                )
            for kind in readout_kinds:
                per_task_cfg = readout_configs_by_task.get(task.name, {})
                cfg = per_task_cfg.get(kind, readout_configs.get(kind))
                if normalize_features and cfg and cfg.get("normalize_features"):
                    cfg = dict(cfg)
                    cfg["normalize_features"] = False
                if y.ndim == 1:
                    reg = _train_single_readout(
                        X, y, readout_kind=kind, readout_config=cfg, rng=rng
                    )
                    metrics = _evaluate_readout(reg, X, y)
                else:
                    models = _train_multi_output(
                        X, y, readout_kind=kind, readout_config=cfg, rng=rng
                    )
                    metrics = _evaluate_multi_output(models, X, y)
                    reg = models[0]
                budget = _budget_info(reg, kind, cfg, X.shape[0])
                results.append(
                    {
                        "task": task.name,
                        "source": source,
                        "readout": kind,
                        "readout_config": dict(cfg or {}),
                        "feature_subsample": int(feature_subsample)
                        if feature_subsample is not None
                        else None,
                        "feature_indices": feature_idx.tolist()
                        if feature_idx is not None
                        else None,
                        **label_info,
                        **budget,
                        **metrics,
                    }
                )
    return results


def write_results_json(results: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_results_csv(results: List[dict], path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _with_seed(kwargs: Dict, seed: int) -> Dict:
    new_kwargs = dict(kwargs)
    if "seed" in new_kwargs:
        new_kwargs["seed"] = int(seed)
    return new_kwargs


def run_seed_sweep(
    task_specs: Iterable[TaskSpec],
    *,
    tasks_to_sweep: Iterable[str] = ("compound_opcode", "serial_adder"),
    seeds: Iterable[int] = range(20),
    readout_config: dict | None = None,
    normalize_features: bool = True,
    feature_subsample: int | None = None,
    output_dir: str = "sweep_outputs",
) -> Dict[str, List[dict]]:
    """Run a seed sweep for EvoReadout and save curves + summaries."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    task_set = set(tasks_to_sweep)
    sweep_results: List[dict] = []
    sweep_curves: List[dict] = []

    evo_config = dict(readout_config) if readout_config is not None else dict(DEFAULT_EVO_CONFIG)

    for task in task_specs:
        if task.name not in task_set:
            continue
        if task.reservoir_builder is None:
            continue

        for seed in seeds:
            kwargs = _with_seed(task.reservoir_kwargs, seed)
            X, y, _meta = task.reservoir_builder(**kwargs)
            label_info = _infer_label_mode(y)
            if label_info["y_mode"] == "multiclass":
                raise ValueError(
                    f"Multiclass labels are not supported yet for task={task.name}"
                )
            if normalize_features:
                X = _zscore_normalize(X)
                if readout_config and readout_config.get("normalize_features"):
                    readout_config = dict(readout_config)
                    readout_config["normalize_features"] = False
            feature_idx = None
            if feature_subsample is not None:
                X, feature_idx = _subsample_features(
                    X, n_features=feature_subsample, rng=np.random.default_rng(seed)
                )

            if y.ndim == 1:
                reg = _train_single_readout(
                    X,
                    y,
                    readout_kind="evo",
                    readout_config=evo_config,
                    rng=np.random.default_rng(seed),
                )
                metrics = _evaluate_readout(reg, X, y)
                curves = {
                    "fitness_best": reg.fitness_history_,
                    "fitness_mean": reg.fitness_mean_history_,
                }
            else:
                models = _train_multi_output(
                    X,
                    y,
                    readout_kind="evo",
                    readout_config=evo_config,
                    rng=np.random.default_rng(seed),
                )
                metrics = _evaluate_multi_output(models, X, y)
                curves = {
                    "fitness_best": [m.fitness_history_ for m in models],
                    "fitness_mean": [m.fitness_mean_history_ for m in models],
                }
                reg = models[0]

            budget = _budget_info(reg, "evo", evo_config, X.shape[0])
            record = {
                "task": task.name,
                "source": "reservoir",
                "readout": "evo",
                "seed": int(seed),
                "readout_config": dict(evo_config),
                "feature_subsample": int(feature_subsample)
                if feature_subsample is not None
                else None,
                "feature_indices": feature_idx.tolist()
                if feature_idx is not None
                else None,
                **label_info,
                **budget,
                **metrics,
            }
            sweep_results.append(record)
            sweep_curves.append(
                {
                    "task": task.name,
                    "seed": int(seed),
                    "source": "reservoir",
                    **curves,
                }
            )

    summary = {}
    for task_name in task_set:
        task_rows = [r for r in sweep_results if r["task"] == task_name]
        if not task_rows:
            continue
        full_acc = np.array([r["full_acc"] for r in task_rows], dtype=float)
        per_bit = np.array([r["per_bit"] for r in task_rows], dtype=float)
        summary[task_name] = {
            "n_seeds": int(len(task_rows)),
            "full_acc_mean": float(full_acc.mean()),
            "full_acc_std": float(full_acc.std(ddof=0)),
            "per_bit_mean": per_bit.mean(axis=0).tolist(),
            "per_bit_std": per_bit.std(axis=0, ddof=0).tolist(),
        }

    results_path = output_path / "sweep_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in sweep_results:
            f.write(json.dumps(row) + "\n")

    curves_path = output_path / "sweep_curves.jsonl"
    with curves_path.open("w", encoding="utf-8") as f:
        for row in sweep_curves:
            f.write(json.dumps(row) + "\n")

    summary_path = output_path / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"results": sweep_results, "summary": summary}


def default_task_specs() -> List[TaskSpec]:
    """Default task list for quick comparisons (edit to suit)."""
    return [
        TaskSpec(
            name="opcode_logic",
            reservoir_builder=build_reservoir_opcode_logic_dataset,
            reservoir_kwargs=dict(
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
            ),
            direct_builder=build_direct_opcode_logic_dataset,
            direct_kwargs={},
        ),
        TaskSpec(
            name="opcode_logic16",
            reservoir_builder=build_reservoir_opcode_logic16_dataset,
            reservoir_kwargs=dict(
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
            ),
            direct_builder=build_direct_opcode_logic16_dataset,
            direct_kwargs={},
        ),
        TaskSpec(
            name="compound_opcode",
            reservoir_builder=build_reservoir_compound_opcode_dataset,
            reservoir_kwargs=dict(
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
            ),
            direct_builder=build_direct_compound_opcode_dataset,
            direct_kwargs={},
        ),
        TaskSpec(
            name="bit_memory",
            reservoir_builder=build_reservoir_bit_memory_dataset,
            reservoir_kwargs=dict(
                bits=6,
                rule_number=110,
                width=256,
                boundary="periodic",
                recurrence=8,
                itr=8,
                d_period=20,
                seed=0,
            ),
            direct_builder=build_direct_bit_memory_dataset,
            direct_kwargs=dict(bits=6),
        ),
        TaskSpec(
            name="serial_adder",
            reservoir_builder=build_reservoir_serial_adder_dataset,
            reservoir_kwargs=dict(
                bits=6,
                n_samples=200,
                rule_number=110,
                width=256,
                boundary="periodic",
                recurrence=8,
                itr=8,
                d_period=3,
                seed=0,
            ),
            direct_builder=build_direct_serial_adder_dataset,
            direct_kwargs=dict(bits=6, n_samples=200, seed=0),
        ),
        TaskSpec(
            name="toy_addition",
            reservoir_builder=build_reservoir_toy_addition_dataset,
            reservoir_kwargs=dict(
                n_bits=3,
                cin=0,
                rule_number=110,
                width=256,
                boundary="periodic",
                itr=8,
                d_period=3,
                recurrence=8,
                repeats=1,
                seed=0,
                feature_mode="cue_tick",
                output_window=2,
            ),
            direct_builder=build_direct_toy_addition_dataset,
            direct_kwargs=dict(n_bits=3, cin=0),
        ),
    ]


def main() -> None:
    results = run_comparison(
        default_task_specs(),
        normalize_features=True,
        feature_subsample=1024,
        readout_configs_by_task={
            "bit_memory": {"evo": EVO_PRESETS["bit_memory"]},
            "compound_opcode": {"evo": EVO_PRESETS["compound_opcode"]},
            "serial_adder": {"evo": EVO_PRESETS["serial_adder"]},
        },
    )
    write_results_json(results, "readout_comparison.json")
    write_results_csv(results, "readout_comparison.csv")


if __name__ == "__main__":
    main()
