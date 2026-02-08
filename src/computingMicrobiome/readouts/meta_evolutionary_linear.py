"""Meta-evolutionary plastic linear readout.

This readout evolves a compact genome that controls how a linear classifier
adapts within its lifetime (Baldwinian-style proxy). The evolved genome is then
used to train final deployable weights on the full dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .base import coerce_xy_binary


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    pos = logits >= 0
    neg = ~pos
    out = np.empty_like(logits, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_x = np.exp(logits[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _binary_log_loss_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    logits_arr = np.asarray(logits, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    loss = np.logaddexp(0.0, logits_arr) - y_arr * logits_arr
    return float(loss.mean())


@dataclass(frozen=True)
class MetaGenome:
    """Evolvable parameters that control within-lifetime adaptation."""

    learning_rate: float
    l2: float
    momentum: float
    init_scale: float
    adaptation_steps: int

    def to_dict(self) -> dict:
        return {
            "learning_rate": float(self.learning_rate),
            "l2": float(self.l2),
            "momentum": float(self.momentum),
            "init_scale": float(self.init_scale),
            "adaptation_steps": int(self.adaptation_steps),
        }


class FrozenLinearReadout:
    """Portable inference-only artifact produced by a fitted meta-evo readout."""

    def __init__(
        self,
        *,
        coef: np.ndarray,
        intercept: float,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        genome: MetaGenome | None = None,
    ) -> None:
        self.coef_ = np.asarray(coef, dtype=float).reshape(-1)
        self.intercept_ = float(intercept)
        self.feature_mean_ = (
            None if feature_mean is None else np.asarray(feature_mean, dtype=float).reshape(1, -1)
        )
        self.feature_std_ = (
            None if feature_std is None else np.asarray(feature_std, dtype=float).reshape(1, -1)
        )
        self.genome_ = genome

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if self.feature_mean_ is not None and self.feature_std_ is not None:
            X_arr = (X_arr - self.feature_mean_) / self.feature_std_
        return X_arr

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_arr = self._transform(X)
        return X_arr @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(np.int8)

    def score(self, X: np.ndarray, y: Iterable[int]) -> float:
        _, y_bin = coerce_xy_binary(X, y)
        y_pred = self.predict(X).reshape(-1)
        return float((y_pred == y_bin.astype(np.int8)).mean())

    def to_dict(self) -> dict:
        return {
            "kind": "frozen_linear_readout_v1",
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
            "feature_mean": None if self.feature_mean_ is None else self.feature_mean_.reshape(-1).tolist(),
            "feature_std": None if self.feature_std_ is None else self.feature_std_.reshape(-1).tolist(),
            "genome": None if self.genome_ is None else self.genome_.to_dict(),
        }

    def save_json(self, path: str | Path) -> Path:
        """Save a frozen readout artifact to JSON."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path_obj

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenLinearReadout":
        kind = payload.get("kind")
        if kind not in {None, "frozen_linear_readout_v1"}:
            raise ValueError(f"Unsupported frozen readout kind: {kind}")
        genome_payload = payload.get("genome")
        genome = None
        if isinstance(genome_payload, Mapping):
            genome = MetaGenome(
                learning_rate=float(genome_payload["learning_rate"]),
                l2=float(genome_payload["l2"]),
                momentum=float(genome_payload["momentum"]),
                init_scale=float(genome_payload["init_scale"]),
                adaptation_steps=int(genome_payload["adaptation_steps"]),
            )
        return cls(
            coef=np.asarray(payload["coef"], dtype=float),
            intercept=float(payload["intercept"]),
            feature_mean=None
            if payload.get("feature_mean") is None
            else np.asarray(payload["feature_mean"], dtype=float),
            feature_std=None
            if payload.get("feature_std") is None
            else np.asarray(payload["feature_std"], dtype=float),
            genome=genome,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "FrozenLinearReadout":
        """Load a frozen readout artifact from JSON."""
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)


class MetaEvolutionaryLinearReadout:
    """Linear readout that meta-evolves a plasticity genome."""

    def __init__(
        self,
        *,
        population_size: int = 64,
        generations: int = 150,
        tournament_size: int = 5,
        elite_count: int = 4,
        mutation_scale: float = 0.2,
        support_fraction: float = 0.6,
        batch_size: int | None = None,
        min_adaptation_steps: int = 2,
        max_adaptation_steps: int = 20,
        adaptation_penalty: float = 0.01,
        lr_bounds: tuple[float, float] = (1e-4, 0.5),
        l2_bounds: tuple[float, float] = (1e-7, 1e-1),
        init_scale_bounds: tuple[float, float] = (1e-3, 0.5),
        momentum_bounds: tuple[float, float] = (0.0, 0.95),
        normalize_features: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_size = int(tournament_size)
        self.elite_count = int(elite_count)
        self.mutation_scale = float(mutation_scale)
        self.support_fraction = float(support_fraction)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.min_adaptation_steps = int(min_adaptation_steps)
        self.max_adaptation_steps = int(max_adaptation_steps)
        self.adaptation_penalty = float(adaptation_penalty)
        self.lr_bounds = (float(lr_bounds[0]), float(lr_bounds[1]))
        self.l2_bounds = (float(l2_bounds[0]), float(l2_bounds[1]))
        self.init_scale_bounds = (float(init_scale_bounds[0]), float(init_scale_bounds[1]))
        self.momentum_bounds = (float(momentum_bounds[0]), float(momentum_bounds[1]))
        self.normalize_features = bool(normalize_features)
        self._rng = rng

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None
        self.best_genome_: MetaGenome | None = None
        self.best_fitness_: float | None = None
        self.fitness_history_: list[float] = []
        self.fitness_mean_history_: list[float] = []
        self.query_loss_history_: list[float] = []
        self.query_acc_history_: list[float] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def _get_rng(self, rng: np.random.Generator | None) -> np.random.Generator:
        if rng is not None:
            return rng
        if self._rng is not None:
            return self._rng
        return np.random.default_rng()

    def _sample_log_uniform(
        self, lo: float, hi: float, rng: np.random.Generator
    ) -> float:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    def _clip_steps(self, value: int) -> int:
        return int(min(max(value, self.min_adaptation_steps), self.max_adaptation_steps))

    def _random_genome(self, rng: np.random.Generator) -> MetaGenome:
        return MetaGenome(
            learning_rate=self._sample_log_uniform(*self.lr_bounds, rng=rng),
            l2=self._sample_log_uniform(*self.l2_bounds, rng=rng),
            momentum=float(rng.uniform(*self.momentum_bounds)),
            init_scale=self._sample_log_uniform(*self.init_scale_bounds, rng=rng),
            adaptation_steps=int(
                rng.integers(self.min_adaptation_steps, self.max_adaptation_steps + 1)
            ),
        )

    def _init_population(self, rng: np.random.Generator) -> list[MetaGenome]:
        pop_size = max(2, self.population_size)
        return [self._random_genome(rng) for _ in range(pop_size)]

    def _mutate_genome(
        self, genome: MetaGenome, rng: np.random.Generator
    ) -> MetaGenome:
        step_span = max(1, self.max_adaptation_steps - self.min_adaptation_steps)

        lr = genome.learning_rate * float(np.exp(rng.normal(0.0, self.mutation_scale)))
        l2 = genome.l2 * float(np.exp(rng.normal(0.0, self.mutation_scale)))
        init_scale = genome.init_scale * float(np.exp(rng.normal(0.0, self.mutation_scale)))
        momentum = genome.momentum + float(rng.normal(0.0, self.mutation_scale * 0.2))
        steps = genome.adaptation_steps + int(
            np.rint(rng.normal(0.0, self.mutation_scale * step_span * 0.5))
        )

        lr = float(np.clip(lr, self.lr_bounds[0], self.lr_bounds[1]))
        l2 = float(np.clip(l2, self.l2_bounds[0], self.l2_bounds[1]))
        init_scale = float(np.clip(init_scale, self.init_scale_bounds[0], self.init_scale_bounds[1]))
        momentum = float(np.clip(momentum, self.momentum_bounds[0], self.momentum_bounds[1]))
        steps = self._clip_steps(steps)

        return MetaGenome(
            learning_rate=lr,
            l2=l2,
            momentum=momentum,
            init_scale=init_scale,
            adaptation_steps=steps,
        )

    def _select_parent_index(
        self, fitness: np.ndarray, rng: np.random.Generator
    ) -> int:
        pop_size = fitness.shape[0]
        t_size = max(2, min(self.tournament_size, pop_size))
        candidates = rng.integers(0, pop_size, size=t_size)
        best = int(candidates[np.argmax(fitness[candidates])])
        return best

    def _split_support_query(
        self, n_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if n_samples <= 1:
            idx = np.arange(n_samples, dtype=int)
            return idx, idx

        n_support = int(round(self.support_fraction * n_samples))
        n_support = max(1, min(n_samples - 1, n_support))
        order = rng.permutation(n_samples)
        support = order[:n_support]
        query = order[n_support:]
        if query.size == 0:
            query = support
        return support, query

    def _adapt_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        genome: MetaGenome,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        n_features = X.shape[1]
        w = rng.normal(0.0, genome.init_scale, size=n_features).astype(float)
        b = float(rng.normal(0.0, genome.init_scale))
        velocity_w = np.zeros_like(w)
        velocity_b = 0.0

        batch_size = self.batch_size
        for _ in range(genome.adaptation_steps):
            if batch_size is None or batch_size >= X.shape[0]:
                X_batch = X
                y_batch = y
            else:
                idx = rng.choice(X.shape[0], size=batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]

            logits = X_batch @ w + b
            probs = _sigmoid(logits)
            err = probs - y_batch
            grad_w = (X_batch.T @ err) / X_batch.shape[0] + genome.l2 * w
            grad_b = float(err.mean())

            velocity_w = genome.momentum * velocity_w + grad_w
            velocity_b = genome.momentum * velocity_b + grad_b
            w -= genome.learning_rate * velocity_w
            b -= genome.learning_rate * velocity_b

        return w, b

    def _evaluate_weights(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
    ) -> tuple[float, float]:
        logits = X @ w + b
        loss = _binary_log_loss_from_logits(logits, y)
        acc = float(((logits >= 0.0).astype(np.int8) == y.astype(np.int8)).mean())
        return loss, acc

    def _next_generation(
        self,
        population: list[MetaGenome],
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> list[MetaGenome]:
        pop_size = len(population)
        elite = max(0, min(self.elite_count, pop_size))
        order = np.argsort(fitness)[::-1]

        new_population: list[MetaGenome] = []
        for idx in order[:elite]:
            new_population.append(population[int(idx)])

        while len(new_population) < pop_size:
            parent_idx = self._select_parent_index(fitness, rng)
            child = self._mutate_genome(population[parent_idx], rng)
            new_population.append(child)

        return new_population

    def fit(
        self,
        X: np.ndarray,
        y: Iterable[int],
        rng: np.random.Generator | None = None,
    ):
        X_arr, y_arr = coerce_xy_binary(X, y)
        if self.normalize_features:
            self._feature_mean = X_arr.mean(axis=0, keepdims=True)
            std = X_arr.std(axis=0, keepdims=True)
            self._feature_std = np.where(std > 0, std, 1.0)
            X_arr = (X_arr - self._feature_mean) / self._feature_std

        rng = self._get_rng(rng)
        self.n_features_in_ = X_arr.shape[1]
        self.fitness_history_ = []
        self.fitness_mean_history_ = []
        self.query_loss_history_ = []
        self.query_acc_history_ = []
        self.best_genome_ = None
        self.best_fitness_ = None

        population = self._init_population(rng)

        for _ in range(self.generations):
            support_idx, query_idx = self._split_support_query(X_arr.shape[0], rng)
            X_support = X_arr[support_idx]
            y_support = y_arr[support_idx]
            X_query = X_arr[query_idx]
            y_query = y_arr[query_idx]

            fitness = np.zeros(len(population), dtype=float)
            losses = np.zeros(len(population), dtype=float)
            accs = np.zeros(len(population), dtype=float)

            for i, genome in enumerate(population):
                w, b = self._adapt_weights(X_support, y_support, genome, rng)
                loss, acc = self._evaluate_weights(X_query, y_query, w, b)
                penalty = self.adaptation_penalty * (
                    genome.adaptation_steps / max(1, self.max_adaptation_steps)
                )
                fit_val = -loss - penalty
                fitness[i] = fit_val
                losses[i] = loss
                accs[i] = acc

            best_idx = int(np.argmax(fitness))
            best_fit = float(fitness[best_idx])
            self.fitness_history_.append(best_fit)
            self.fitness_mean_history_.append(float(fitness.mean()))
            self.query_loss_history_.append(float(losses[best_idx]))
            self.query_acc_history_.append(float(accs[best_idx]))

            if self.best_fitness_ is None or best_fit > self.best_fitness_:
                self.best_fitness_ = best_fit
                self.best_genome_ = population[best_idx]

            population = self._next_generation(population, fitness, rng)

        if self.best_genome_ is None:
            raise RuntimeError("Meta-evolution failed to produce a best genome.")

        final_w, final_b = self._adapt_weights(X_arr, y_arr, self.best_genome_, rng)
        self.coef_ = final_w.astype(float)
        self.intercept_ = float(final_b)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Readout not fitted.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if self.normalize_features and self._feature_mean is not None:
            X_arr = (X_arr - self._feature_mean) / self._feature_std
        return X_arr @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(np.int8)

    def score(self, X: np.ndarray, y: Iterable[int]) -> float:
        y_pred = self.predict(X).reshape(-1)
        _, y_bin = coerce_xy_binary(X, y)
        return float((y_pred == y_bin.astype(np.int8)).mean())

    def freeze(self) -> FrozenLinearReadout:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Readout not fitted.")
        return FrozenLinearReadout(
            coef=self.coef_,
            intercept=self.intercept_,
            feature_mean=self._feature_mean,
            feature_std=self._feature_std,
            genome=self.best_genome_,
        )

    def save_frozen_json(self, path: str | Path) -> Path:
        """Freeze the readout and save it as a JSON artifact."""
        return self.freeze().save_json(path)
