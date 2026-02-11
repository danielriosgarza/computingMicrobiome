"""Microbiome-centric host evolution with fixed-capacity learners.

This module implements a simpler outer-loop evolutionary process aligned with
the host-microbiome framing:

- Inner loop: every host uses the same learner class/capacity and calls fit().
- Outer loop: hosts reproduce with probability proportional to task fitness.
- Heritable unit: microbiome proxy parameters (ECA reservoir settings), not
  learner hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Callable, Dict, Iterable, List, Mapping

import numpy as np
from sklearn.linear_model import SGDClassifier

from ..feature_sources import (
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
from ..readouts.factory import make_readout
from .base import GenerationMetrics
from .results import EvolutionRunResult

DatasetBuilder = Callable[..., tuple[np.ndarray, np.ndarray, Dict[str, Any]]]


_TASK_BUILDERS: Dict[str, Dict[str, DatasetBuilder]] = {
    "bit_memory": {
        "direct": build_direct_bit_memory_dataset,
        "reservoir": build_reservoir_bit_memory_dataset,
    },
    "opcode_logic": {
        "direct": build_direct_opcode_logic_dataset,
        "reservoir": build_reservoir_opcode_logic_dataset,
    },
    "opcode_logic16": {
        "direct": build_direct_opcode_logic16_dataset,
        "reservoir": build_reservoir_opcode_logic16_dataset,
    },
    "compound_opcode": {
        "direct": build_direct_compound_opcode_dataset,
        "reservoir": build_reservoir_compound_opcode_dataset,
    },
    "serial_adder": {
        "direct": build_direct_serial_adder_dataset,
        "reservoir": build_reservoir_serial_adder_dataset,
    },
    "toy_addition": {
        "direct": build_direct_toy_addition_dataset,
        "reservoir": build_reservoir_toy_addition_dataset,
    },
}


@dataclass
class MicrobiomeIndividual:
    """Heritable microbiome proxy parameters for one host."""

    source: str
    rule_number: int | None
    seed: int


def _filter_kwargs_for_callable(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict:
    params = signature(fn).parameters
    if any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _get_builder(task: str, source: str) -> DatasetBuilder:
    task_key = task.strip().lower()
    source_key = source.strip().lower()
    if task_key not in _TASK_BUILDERS:
        raise ValueError(
            f"Unknown task: {task!r}. Valid tasks: {sorted(_TASK_BUILDERS.keys())}"
        )
    if source_key not in _TASK_BUILDERS[task_key]:
        raise ValueError("source must be 'direct' or 'reservoir'")
    return _TASK_BUILDERS[task_key][source_key]


def _build_dataset_for_individual(
    task: str,
    base_kwargs: Mapping[str, Any],
    ind: MicrobiomeIndividual,
) -> tuple[np.ndarray, np.ndarray]:
    builder = _get_builder(task, ind.source)
    kwargs = dict(base_kwargs)
    if ind.source == "reservoir":
        kwargs["seed"] = int(ind.seed)
        if ind.rule_number is not None:
            kwargs["rule_number"] = int(ind.rule_number)
    else:
        # Some direct builders accept a seed (e.g., serial adder); others do not.
        kwargs["seed"] = int(ind.seed)

    filtered = _filter_kwargs_for_callable(builder, kwargs)
    X, y, _meta = builder(**filtered)
    return np.asarray(X), np.asarray(y)


def _freeze_for_key(value: Any) -> Any:
    """Convert nested objects to hashable, deterministic structures."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze_for_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_key(v) for v in value)
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), tuple(value.shape), tuple(value.reshape(-1).tolist()))
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _build_dataset_cache_key(
    *,
    task: str,
    base_kwargs: Mapping[str, Any],
    ind: MicrobiomeIndividual,
) -> tuple[Any, ...]:
    builder = _get_builder(task, ind.source)
    kwargs = dict(base_kwargs)
    if ind.source == "reservoir":
        kwargs["seed"] = int(ind.seed)
        if ind.rule_number is not None:
            kwargs["rule_number"] = int(ind.rule_number)
    else:
        kwargs["seed"] = int(ind.seed)
    filtered = _filter_kwargs_for_callable(builder, kwargs)
    return (
        task,
        ind.source,
        _freeze_for_key(filtered),
    )


def _sample_rows(
    X: np.ndarray,
    y: np.ndarray,
    n_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")
    n = X.shape[0]
    take = min(n_rows, n)
    idx = rng.choice(n, size=take, replace=False)
    return X[idx], y[idx]


def _fit_and_score_binary(
    X_sup: np.ndarray,
    y_sup: np.ndarray,
    X_ch: np.ndarray,
    y_ch: np.ndarray,
    *,
    learner_kind: str,
    learner_config: dict | None,
    rng: np.random.Generator,
) -> float:
    y_unique = np.unique(np.asarray(y_sup).reshape(-1))
    if y_unique.size < 2:
        const = int(y_unique[0])
        y_pred = np.full(np.asarray(y_ch).shape[0], const, dtype=np.int8)
        y_true = np.asarray(y_ch).reshape(-1).astype(np.int8)
        return float((y_pred == y_true).mean())

    reg = make_readout(learner_kind, learner_config, rng=rng)
    reg.fit(X_sup, y_sup)
    return float(reg.score(X_ch, y_ch))


def _fit_and_score_multilabel(
    X_sup: np.ndarray,
    Y_sup: np.ndarray,
    X_ch: np.ndarray,
    Y_ch: np.ndarray,
    *,
    learner_kind: str,
    learner_config: dict | None,
    fitness_metric: str,
    rng: np.random.Generator,
) -> float:
    preds = np.zeros_like(Y_ch, dtype=np.int8)
    for col in range(Y_sup.shape[1]):
        y_sup_col = np.asarray(Y_sup[:, col]).reshape(-1)
        y_unique = np.unique(y_sup_col)
        if y_unique.size < 2:
            preds[:, col] = int(y_unique[0])
        else:
            reg = make_readout(learner_kind, learner_config, rng=rng)
            reg.fit(X_sup, y_sup_col)
            preds[:, col] = reg.predict(X_ch).astype(np.int8)

    if fitness_metric == "mean_bit_accuracy":
        return float((preds == Y_ch).mean())
    if fitness_metric == "full_vector_accuracy":
        return float((preds == Y_ch).all(axis=1).mean())
    raise ValueError(
        "fitness_metric must be one of: 'mean_bit_accuracy', 'full_vector_accuracy'"
    )


def _fit_and_score_mistake_driven_binary(
    X_sup: np.ndarray,
    y_sup: np.ndarray,
    X_ch: np.ndarray,
    y_ch: np.ndarray,
    *,
    learner_config: dict | None,
    online_batch_size: int,
    rng: np.random.Generator,
) -> float:
    if online_batch_size < 1:
        raise ValueError("online_batch_size must be >= 1")

    cfg = dict(learner_config or {})
    alpha = float(cfg.get("alpha", 1e-4))
    penalty = str(cfg.get("penalty", "l2"))
    fit_intercept = bool(cfg.get("fit_intercept", True))

    clf = SGDClassifier(
        loss="hinge",
        alpha=alpha,
        penalty=penalty,
        fit_intercept=fit_intercept,
        random_state=int(rng.integers(0, 2**31 - 1)),
        max_iter=1,
        tol=None,
    )

    X_sup_arr = np.asarray(X_sup, dtype=float)
    y_sup_arr = np.asarray(y_sup).reshape(-1).astype(np.int8)
    X_ch_arr = np.asarray(X_ch, dtype=float)
    y_ch_arr = np.asarray(y_ch).reshape(-1).astype(np.int8)

    # Initial mini-batch fit: seed the learner with first experience.
    clf.partial_fit(X_sup_arr, y_sup_arr, classes=np.array([0, 1], dtype=np.int8))

    correct = 0
    n = int(y_ch_arr.shape[0])
    for start in range(0, n, online_batch_size):
        stop = min(start + online_batch_size, n)
        Xb = X_ch_arr[start:stop]
        yb = y_ch_arr[start:stop]

        # Prequential score: predict first, then update only on mistakes.
        y_hat = clf.predict(Xb).astype(np.int8)
        correct += int((y_hat == yb).sum())

        mistakes = y_hat != yb
        if mistakes.any():
            clf.partial_fit(Xb[mistakes], yb[mistakes])

    return float(correct / max(1, n))


def _evaluate_individual(
    X: np.ndarray,
    y: np.ndarray,
    *,
    support_size: int,
    challenge_size: int,
    learner_kind: str,
    learner_config: dict | None,
    fitness_metric: str,
    incremental_mistake_updates: bool,
    online_batch_size: int,
    rng: np.random.Generator,
    shared_challenge_idx: np.ndarray | None = None,
) -> float:
    X_sup, y_sup = _sample_rows(X, y, support_size, rng)

    if shared_challenge_idx is not None:
        X_ch = X[shared_challenge_idx]
        y_ch = y[shared_challenge_idx]
    else:
        X_ch, y_ch = _sample_rows(X, y, challenge_size, rng)

    if y.ndim == 1:
        if incremental_mistake_updates:
            return _fit_and_score_mistake_driven_binary(
                X_sup,
                y_sup,
                X_ch,
                y_ch,
                learner_config=learner_config,
                online_batch_size=online_batch_size,
                rng=rng,
            )
        return _fit_and_score_binary(
            X_sup,
            y_sup,
            X_ch,
            y_ch,
            learner_kind=learner_kind,
            learner_config=learner_config,
            rng=rng,
        )
    if y.ndim == 2:
        if incremental_mistake_updates:
            raise ValueError(
                "incremental_mistake_updates currently supports only binary (1D) targets"
            )
        return _fit_and_score_multilabel(
            X_sup,
            y_sup,
            X_ch,
            y_ch,
            learner_kind=learner_kind,
            learner_config=learner_config,
            fitness_metric=fitness_metric,
            rng=rng,
        )
    raise ValueError("Only binary (1D) and multi-label (2D) y are supported.")


def _fitness_proportional_weights(fitness: np.ndarray) -> np.ndarray:
    fit = np.asarray(fitness, dtype=float)
    shifted = fit - float(fit.min()) + 1e-12
    total = float(shifted.sum())
    if total <= 0.0:
        return np.full(fit.shape[0], 1.0 / fit.shape[0], dtype=float)
    return shifted / total


def _mutate_individual(
    ind: MicrobiomeIndividual,
    *,
    mutate_rule_prob: float,
    mutate_seed_prob: float,
    rng: np.random.Generator,
) -> MicrobiomeIndividual:
    child = MicrobiomeIndividual(source=ind.source, rule_number=ind.rule_number, seed=int(ind.seed))

    if ind.source == "reservoir" and child.rule_number is not None:
        if rng.random() < mutate_rule_prob:
            step = int(rng.choice([-1, 1]))
            child.rule_number = int((child.rule_number + step) % 256)

    if rng.random() < mutate_seed_prob:
        child.seed = int(rng.integers(0, 2**31 - 1))

    return child


def _make_generation_iterator(
    generations: int,
    *,
    progress: bool,
    progress_mode: str,
    progress_desc: str,
) -> tuple[Iterable[int], object | None, bool]:
    """Build generation iterator with optional progress UI."""
    if not progress:
        return range(generations), None, False

    mode = progress_mode.strip().lower()
    if mode not in {"auto", "tqdm", "print"}:
        raise ValueError("progress_mode must be one of: auto, tqdm, print")

    if mode in {"auto", "tqdm"}:
        try:
            from tqdm.auto import tqdm  # type: ignore

            bar = tqdm(range(generations), total=generations, desc=progress_desc)
            return bar, bar, False
        except Exception:
            if mode == "tqdm":
                raise
            return range(generations), None, True

    return range(generations), None, True


def _normalize_initial_rules(
    *,
    use_reservoir: bool,
    base_kwargs: Mapping[str, Any],
    rules: Iterable[int] | None,
) -> list[int] | None:
    if not use_reservoir:
        if rules is not None:
            raise ValueError("rules can only be provided when using reservoir sources")
        return None

    if rules is None:
        return [int(base_kwargs.get("rule_number", 110)) % 256]

    normalized = [int(r) % 256 for r in rules]
    if not normalized:
        raise ValueError("rules must contain at least one rule number when provided")
    return normalized


def _normalize_initial_sources(*, source: str, sources: Iterable[str] | None) -> list[str]:
    if sources is None:
        return [source]
    normalized = [str(s).strip().lower() for s in sources if str(s).strip()]
    if not normalized:
        raise ValueError("sources must contain at least one value when provided")
    for s in normalized:
        if s not in {"direct", "reservoir"}:
            raise ValueError("sources values must be 'direct' or 'reservoir'")
    return normalized


def _build_initial_population(
    *,
    population_size: int,
    init_sources: list[str],
    init_rules: list[int] | None,
    rng: np.random.Generator,
) -> list[MicrobiomeIndividual]:
    repeated_sources = [init_sources[i % len(init_sources)] for i in range(population_size)]
    rng.shuffle(repeated_sources)

    population: list[MicrobiomeIndividual] = []
    reservoir_count = 0
    for src in repeated_sources:
        rule = None
        if src == "reservoir":
            if not init_rules:
                raise ValueError("Reservoir initialization requires at least one rule")
            rule = int(init_rules[reservoir_count % len(init_rules)])
            reservoir_count += 1
        population.append(
            MicrobiomeIndividual(
                source=src,
                rule_number=rule,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return population


def _population_composition(population: list[MicrobiomeIndividual]) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    by_source_rule: dict[str, dict[str, int]] = {}
    for ind in population:
        src = str(ind.source)
        by_source[src] = by_source.get(src, 0) + 1
        src_rules = by_source_rule.setdefault(src, {})
        if ind.rule_number is None:
            continue
        rule_key = str(int(ind.rule_number))
        by_rule[rule_key] = by_rule.get(rule_key, 0) + 1
        src_rules[rule_key] = src_rules.get(rule_key, 0) + 1
    return {
        "by_source": dict(sorted(by_source.items())),
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: int(kv[0]))),
        "by_source_rule": {
            src: dict(sorted(rules.items(), key=lambda kv: int(kv[0])))
            for src, rules in sorted(by_source_rule.items())
        },
    }


def _serialize_population(population: list[MicrobiomeIndividual]) -> list[dict[str, Any]]:
    return [
        {
            "source": ind.source,
            "rule_number": None if ind.rule_number is None else int(ind.rule_number),
            "seed": int(ind.seed),
        }
        for ind in population
    ]


def run_microbiome_host_evolution(
    *,
    task: str,
    source: str,
    sources: Iterable[str] | None = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    rules: Iterable[int] | None = None,
    learner_kind: str = "svm",
    learner_config: Mapping[str, Any] | None = None,
    population_size: int = 64,
    generations: int = 100,
    support_size: int = 128,
    challenge_size: int = 128,
    fitness_metric: str = "full_vector_accuracy",
    incremental_mistake_updates: bool = False,
    online_batch_size: int = 12,
    shared_challenge_across_population: bool = True,
    mutate_rule_prob: float = 0.0,
    mutate_seed_prob: float = 0.2,
    seed: int = 0,
    progress: bool = False,
    progress_mode: str = "auto",
    progress_every: int = 10,
    progress_desc: str = "Microbiome Evolution",
    inner_progress: bool = True,
    inner_progress_every: int = 5,
    use_dataset_cache: bool = True,
) -> EvolutionRunResult:
    """Run host evolution with fixed-capacity learners and microbiome inheritance.

    Args:
        task: One of
            {"bit_memory", "opcode_logic", "opcode_logic16", "compound_opcode",
             "serial_adder", "toy_addition"}.
        source: Backward-compatible default source when sources is not provided.
            One of {"direct", "reservoir"}.
        sources: Optional initial sources to mix in one population. Values must be
            from {"direct", "reservoir"}. Hosts are initialized in near-equal counts.
        dataset_kwargs: Task/builder kwargs (e.g. bits, n_samples, rule_number, ...).
        rules: Optional initial reservoir rules. When provided, initial hosts are
            assigned these rule values in near-equal counts across the population.
        learner_kind: Readout backend from readout factory (svm, evo, moran, ...).
        learner_config: Config passed to readout factory.
        population_size: Number of hosts in the population.
        generations: Number of evolutionary generations.
        support_size: Training rows sampled per host per generation.
        challenge_size: Evaluation rows sampled per host (or shared challenge size).
        fitness_metric: For 2D labels, "full_vector_accuracy" or "mean_bit_accuracy".
        incremental_mistake_updates: If True, use prequential evaluation on challenge
            rows and update only on prediction mistakes after an initial support fit.
            Applies to binary tasks only.
        online_batch_size: Challenge stream batch size for mistake-driven updates.
        shared_challenge_across_population: If True, all hosts use same challenge rows.
        mutate_rule_prob: Per-offspring mutation probability for reservoir rule number.
        mutate_seed_prob: Per-offspring mutation probability for dataset seed.
        seed: Global RNG seed.
        progress: If True, show run progress.
        progress_mode: One of {"auto", "tqdm", "print"}.
        progress_every: Generation interval for text progress when using print mode.
        progress_desc: Progress bar label when tqdm mode is used.
        inner_progress: If True, show per-host progress within each generation.
        inner_progress_every: Host interval for text updates in print mode.
        use_dataset_cache: If True, memoize built datasets by genotype/config.

    Returns:
        EvolutionRunResult: Generation history and final population summary.
    """
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if generations < 1:
        raise ValueError("generations must be >= 1")
    if support_size < 1 or challenge_size < 1:
        raise ValueError("support_size and challenge_size must be >= 1")
    if online_batch_size < 1:
        raise ValueError("online_batch_size must be >= 1")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")
    if inner_progress_every < 1:
        raise ValueError("inner_progress_every must be >= 1")
    if source not in {"direct", "reservoir"}:
        raise ValueError("source must be 'direct' or 'reservoir'")

    base_kwargs = dict(dataset_kwargs or {})
    l_cfg = dict(learner_config or {})

    rng = np.random.default_rng(seed)
    init_sources = _normalize_initial_sources(source=source, sources=sources)
    init_rules = _normalize_initial_rules(
        use_reservoir=("reservoir" in init_sources),
        base_kwargs=base_kwargs,
        rules=rules,
    )
    population = _build_initial_population(
        population_size=population_size,
        init_sources=init_sources,
        init_rules=init_rules,
        rng=rng,
    )

    history: List[GenerationMetrics] = []
    composition_history: List[dict[str, Any]] = []
    final_fitness = np.zeros(population_size, dtype=float)
    dataset_cache: Dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    cache_hits = 0
    cache_misses = 0

    iterator, progress_bar, use_print_progress = _make_generation_iterator(
        generations,
        progress=progress,
        progress_mode=progress_mode,
        progress_desc=progress_desc,
    )

    for gen in iterator:
        composition_history.append(
            {"generation": int(gen), **_population_composition(population)}
        )
        fitness = np.zeros(population_size, dtype=float)

        # Optional shared challenge indices per generation (for comparability).
        shared_idx_by_source: dict[str, np.ndarray] = {}
        if shared_challenge_across_population:
            for src in sorted({ind.source for ind in population}):
                rep = next(ind for ind in population if ind.source == src)
                key0 = _build_dataset_cache_key(
                    task=task,
                    base_kwargs=base_kwargs,
                    ind=rep,
                )
                if use_dataset_cache and key0 in dataset_cache:
                    X0, y0 = dataset_cache[key0]
                    cache_hits += 1
                else:
                    X0, y0 = _build_dataset_for_individual(task, base_kwargs, rep)
                    if use_dataset_cache:
                        dataset_cache[key0] = (X0, y0)
                    cache_misses += 1
                challenge_take = min(challenge_size, X0.shape[0])
                shared_idx_by_source[src] = rng.choice(
                    X0.shape[0], size=challenge_take, replace=False
                )

        for i, ind in enumerate(population):
            key = _build_dataset_cache_key(
                task=task,
                base_kwargs=base_kwargs,
                ind=ind,
            )
            if use_dataset_cache and key in dataset_cache:
                X, y = dataset_cache[key]
                cache_hits += 1
            else:
                X, y = _build_dataset_for_individual(task, base_kwargs, ind)
                if use_dataset_cache:
                    dataset_cache[key] = (X, y)
                cache_misses += 1
            fitness[i] = _evaluate_individual(
                X,
                y,
                support_size=support_size,
                challenge_size=challenge_size,
                learner_kind=learner_kind,
                learner_config=l_cfg,
                fitness_metric=fitness_metric,
                incremental_mistake_updates=incremental_mistake_updates,
                online_batch_size=online_batch_size,
                rng=rng,
                shared_challenge_idx=shared_idx_by_source.get(ind.source),
            )
            if inner_progress:
                if progress_bar is not None:
                    total_cache = cache_hits + cache_misses
                    cache_rate = cache_hits / total_cache if total_cache > 0 else 0.0
                    progress_bar.set_postfix(
                        gen=f"{gen + 1}/{generations}",
                        host=f"{i + 1}/{population_size}",
                        cache=f"{cache_rate:.2f}",
                    )
                elif use_print_progress and (
                    (i + 1) % inner_progress_every == 0
                    or i == 0
                    or i == population_size - 1
                ):
                    print(
                        f"[microbiome] gen {gen + 1}/{generations} "
                        f"host {i + 1}/{population_size}"
                    )

        final_fitness = fitness.copy()

        weights = _fitness_proportional_weights(fitness)
        parent_idx = rng.choice(population_size, size=population_size, replace=True, p=weights)

        new_population: List[MicrobiomeIndividual] = []
        for pidx in parent_idx:
            child = _mutate_individual(
                population[int(pidx)],
                mutate_rule_prob=mutate_rule_prob,
                mutate_seed_prob=mutate_seed_prob,
                rng=rng,
            )
            new_population.append(child)

        replacement_rate = float(np.mean(parent_idx != np.arange(population_size)))
        history.append(
            GenerationMetrics(
                generation=gen,
                mean_fitness=float(fitness.mean()),
                best_fitness=float(fitness.max()),
                std_fitness=float(fitness.std()),
                mean_adaptation_gain=0.0,
                replacement_rate=replacement_rate,
            )
        )

        if progress_bar is not None:
            total_cache = cache_hits + cache_misses
            cache_rate = cache_hits / total_cache if total_cache > 0 else 0.0
            progress_bar.set_postfix(
                mean=f"{history[-1].mean_fitness:.3f}",
                best=f"{history[-1].best_fitness:.3f}",
                repl=f"{history[-1].replacement_rate:.2f}",
                cache=f"{cache_rate:.2f}",
            )
        elif use_print_progress and (
            (gen + 1) % progress_every == 0 or gen == 0 or gen == generations - 1
        ):
            print(
                f"[microbiome] gen {gen + 1}/{generations} "
                f"mean={history[-1].mean_fitness:.4f} "
                f"best={history[-1].best_fitness:.4f} "
                f"repl={history[-1].replacement_rate:.3f}"
            )
        population = new_population

    if progress_bar is not None:
        progress_bar.close()

    best_idx = int(np.argmax(final_fitness))
    best = population[best_idx]

    return EvolutionRunResult(
        config={
            "task": task,
            "source": source,
            "sources": list(init_sources),
            "dataset_kwargs": base_kwargs,
            "rules": list(init_rules) if init_rules is not None else None,
            "learner_kind": learner_kind,
            "learner_config": l_cfg,
            "population_size": population_size,
            "generations": generations,
            "support_size": support_size,
            "challenge_size": challenge_size,
            "fitness_metric": fitness_metric,
            "incremental_mistake_updates": incremental_mistake_updates,
            "online_batch_size": online_batch_size,
            "shared_challenge_across_population": shared_challenge_across_population,
            "mutate_rule_prob": mutate_rule_prob,
            "mutate_seed_prob": mutate_seed_prob,
        },
        history=history,
        final_population_fitness=final_fitness.astype(float),
        best_genotype={
            "source": best.source,
            "rule_number": None if best.rule_number is None else int(best.rule_number),
            "seed": int(best.seed),
        },
        seed=seed,
        final_population_genotypes=_serialize_population(population),
        population_composition_history=composition_history,
    )


__all__ = ["MicrobiomeIndividual", "run_microbiome_host_evolution"]
