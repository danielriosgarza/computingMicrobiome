"""Wright-Fisher evolutionary linear readout."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .base import coerce_xy_binary


class EvolutionaryLinearReadout:
    """Linear readout trained with a Wright-Fisher evolutionary process."""

    def __init__(
        self,
        *,
        population_size: int = 128,
        generations: int = 200,
        tournament_size: int = 5,
        elite_count: int = 2,
        mutation_scale: float = 0.1,
        min_mutation_scale: float = 0.0,
        mutation_schedule: str = "constant",
        mutation_fraction: float = 1.0,
        init_scale: float = 0.1,
        batch_size: int | None = None,
        normalize_features: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_size = int(tournament_size)
        self.elite_count = int(elite_count)
        self.mutation_scale = float(mutation_scale)
        self.min_mutation_scale = float(min_mutation_scale)
        self.mutation_schedule = str(mutation_schedule)
        self.mutation_fraction = float(mutation_fraction)
        self.init_scale = float(init_scale)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.normalize_features = bool(normalize_features)
        self._rng = rng

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.fitness_history_: list[float] = []
        self.fitness_mean_history_: list[float] = []
        self.n_features_in_: int | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def _get_rng(self, rng: np.random.Generator | None) -> np.random.Generator:
        if rng is not None:
            return rng
        if self._rng is not None:
            return self._rng
        return np.random.default_rng()

    def _sample_batch(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.batch_size is None or self.batch_size >= X.shape[0]:
            return X, y
        idx = rng.choice(X.shape[0], size=self.batch_size, replace=False)
        return X[idx], y[idx]

    @staticmethod
    def _fitness(
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        logits = X @ weights.T + bias
        loss = np.logaddexp(0.0, logits) - y[:, None] * logits
        return -loss.mean(axis=0)

    def _select_parents(
        self, fitness: np.ndarray, n_parents: int, rng: np.random.Generator
    ) -> np.ndarray:
        pop_size = fitness.shape[0]
        t_size = max(2, min(self.tournament_size, pop_size))
        candidates = rng.integers(0, pop_size, size=(n_parents, t_size))
        best_idx = np.argmax(fitness[candidates], axis=1)
        return candidates[np.arange(n_parents), best_idx]

    def fit(self, X: np.ndarray, y: Iterable[int], rng: np.random.Generator | None = None):
        X_arr, y_arr = coerce_xy_binary(X, y)
        if self.normalize_features:
            self._feature_mean = X_arr.mean(axis=0, keepdims=True)
            std = X_arr.std(axis=0, keepdims=True)
            self._feature_std = np.where(std > 0, std, 1.0)
            X_arr = (X_arr - self._feature_mean) / self._feature_std
        rng = self._get_rng(rng)

        pop_size = max(2, self.population_size)
        elite = max(0, min(self.elite_count, pop_size))
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        weights = rng.normal(0.0, self.init_scale, size=(pop_size, n_features))
        bias = rng.normal(0.0, self.init_scale, size=(pop_size,))

        for gen_idx in range(self.generations):
            X_batch, y_batch = self._sample_batch(X_arr, y_arr, rng)
            fitness = self._fitness(X_batch, y_batch, weights, bias)
            self.fitness_history_.append(float(fitness.max()))
            self.fitness_mean_history_.append(float(fitness.mean()))

            order = np.argsort(fitness)[::-1]
            new_w = np.empty_like(weights)
            new_b = np.empty_like(bias)

            if elite > 0:
                elites = order[:elite]
                new_w[:elite] = weights[elites]
                new_b[:elite] = bias[elites]

            n_offspring = pop_size - elite
            if n_offspring > 0:
                parent_idx = self._select_parents(fitness, n_offspring, rng)
                offspring_w = weights[parent_idx].copy()
                offspring_b = bias[parent_idx].copy()
                mutation_scale = self._mutation_scale_for_gen(gen_idx)
                if mutation_scale > 0:
                    w_noise = rng.normal(0.0, mutation_scale, size=offspring_w.shape)
                    b_noise = rng.normal(0.0, mutation_scale, size=offspring_b.shape)
                    if self.mutation_fraction < 1.0:
                        mask = rng.random(size=offspring_w.shape) < self.mutation_fraction
                        w_noise = w_noise * mask
                        b_mask = rng.random(size=offspring_b.shape) < self.mutation_fraction
                        b_noise = b_noise * b_mask
                    offspring_w += w_noise
                    offspring_b += b_noise
                new_w[elite:] = offspring_w
                new_b[elite:] = offspring_b

            weights, bias = new_w, new_b

        final_fitness = self._fitness(X_arr, y_arr, weights, bias)
        best = int(np.argmax(final_fitness))
        self.coef_ = weights[best].astype(float)
        self.intercept_ = float(bias[best])
        return self

    def _mutation_scale_for_gen(self, gen_idx: int) -> float:
        if self.mutation_schedule == "constant":
            return self.mutation_scale
        if self.mutation_schedule == "linear_decay":
            if self.generations <= 1:
                return max(self.min_mutation_scale, self.mutation_scale)
            frac = 1.0 - (gen_idx / (self.generations - 1))
            scaled = self.mutation_scale * frac
            return max(self.min_mutation_scale, scaled)
        raise ValueError(f"Unknown mutation_schedule: {self.mutation_schedule}")

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
        logits = self.decision_function(X)
        return (logits >= 0.0).astype(np.int8)

    def score(self, X: np.ndarray, y: Iterable[int]) -> float:
        y_pred = self.predict(X).reshape(-1)
        _, y_bin = coerce_xy_binary(X, y)
        return float((y_pred == y_bin.astype(np.int8)).mean())
