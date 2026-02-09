"""Conservative Moran-style evolutionary linear readout."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .base import coerce_xy_binary


class MoranLinearReadout:
    """Linear readout trained with a steady-state Moran-style process.

    Unlike full generational replacement, each evolutionary step proposes a
    single mutated offspring and performs at most one replacement in the
    population. This conservative update can be more stable in high dimensions.
    """

    def __init__(
        self,
        *,
        population_size: int = 128,
        generations: int = 1000,
        tournament_size: int = 5,
        death_tournament_size: int = 5,
        mutation_scale: float = 0.05,
        min_mutation_scale: float = 0.0,
        mutation_schedule: str = "constant",
        mutation_fraction: float = 0.1,
        init_scale: float = 0.1,
        batch_size: int | None = None,
        accept_only_improving: bool = True,
        normalize_features: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_size = int(tournament_size)
        self.death_tournament_size = int(death_tournament_size)
        self.mutation_scale = float(mutation_scale)
        self.min_mutation_scale = float(min_mutation_scale)
        self.mutation_schedule = str(mutation_schedule)
        self.mutation_fraction = float(mutation_fraction)
        self.init_scale = float(init_scale)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.accept_only_improving = bool(accept_only_improving)
        self.normalize_features = bool(normalize_features)
        self._rng = rng

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.fitness_history_: list[float] = []
        self.fitness_mean_history_: list[float] = []
        self.replacement_rate_history_: list[float] = []
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

    @staticmethod
    def _fitness_single(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        logits = X @ w + b
        loss = np.logaddexp(0.0, logits) - y * logits
        return float(-loss.mean())

    def _select_parent_index(
        self, fitness: np.ndarray, rng: np.random.Generator
    ) -> int:
        pop_size = fitness.shape[0]
        t_size = max(2, min(self.tournament_size, pop_size))
        candidates = rng.integers(0, pop_size, size=t_size)
        return int(candidates[np.argmax(fitness[candidates])])

    def _select_death_index(
        self, fitness: np.ndarray, rng: np.random.Generator
    ) -> int:
        pop_size = fitness.shape[0]
        t_size = max(2, min(self.death_tournament_size, pop_size))
        candidates = rng.integers(0, pop_size, size=t_size)
        return int(candidates[np.argmin(fitness[candidates])])

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

    def _mutate(
        self,
        w: np.ndarray,
        b: float,
        mutation_scale: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        child_w = w.copy()
        child_b = float(b)

        if mutation_scale <= 0:
            return child_w, child_b

        w_noise = rng.normal(0.0, mutation_scale, size=child_w.shape)
        if self.mutation_fraction < 1.0:
            mask = rng.random(size=child_w.shape) < self.mutation_fraction
            w_noise = w_noise * mask
            if rng.random() >= self.mutation_fraction:
                b_noise = 0.0
            else:
                b_noise = float(rng.normal(0.0, mutation_scale))
        else:
            b_noise = float(rng.normal(0.0, mutation_scale))

        child_w += w_noise
        child_b += b_noise
        return child_w, child_b

    def fit(
        self, X: np.ndarray, y: Iterable[int], rng: np.random.Generator | None = None
    ):
        X_arr, y_arr = coerce_xy_binary(X, y)
        if self.normalize_features:
            self._feature_mean = X_arr.mean(axis=0, keepdims=True)
            std = X_arr.std(axis=0, keepdims=True)
            self._feature_std = np.where(std > 0, std, 1.0)
            X_arr = (X_arr - self._feature_mean) / self._feature_std
        rng = self._get_rng(rng)

        pop_size = max(2, self.population_size)
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        weights = rng.normal(0.0, self.init_scale, size=(pop_size, n_features))
        bias = rng.normal(0.0, self.init_scale, size=(pop_size,))

        replacements = 0
        self.fitness_history_ = []
        self.fitness_mean_history_ = []
        self.replacement_rate_history_ = []

        for gen_idx in range(self.generations):
            X_batch, y_batch = self._sample_batch(X_arr, y_arr, rng)

            # Recompute fitness on batch for current generation dynamics.
            fitness_batch = self._fitness(X_batch, y_batch, weights, bias)
            self.fitness_history_.append(float(fitness_batch.max()))
            self.fitness_mean_history_.append(float(fitness_batch.mean()))

            parent_idx = self._select_parent_index(fitness_batch, rng)
            death_idx = self._select_death_index(fitness_batch, rng)

            mutation_scale = self._mutation_scale_for_gen(gen_idx)
            child_w, child_b = self._mutate(
                weights[parent_idx], bias[parent_idx], mutation_scale, rng
            )
            child_fit_batch = self._fitness_single(X_batch, y_batch, child_w, child_b)

            should_replace = True
            if self.accept_only_improving:
                should_replace = child_fit_batch > float(fitness_batch[death_idx])

            if should_replace:
                weights[death_idx] = child_w
                bias[death_idx] = child_b
                replacements += 1

            self.replacement_rate_history_.append(replacements / float(gen_idx + 1))

        fitness_full = self._fitness(X_arr, y_arr, weights, bias)
        best_idx = int(np.argmax(fitness_full))
        self.coef_ = weights[best_idx].astype(float)
        self.intercept_ = float(bias[best_idx])
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
        logits = self.decision_function(X)
        return (logits >= 0.0).astype(np.int8)

    def score(self, X: np.ndarray, y: Iterable[int]) -> float:
        y_pred = self.predict(X).reshape(-1)
        _, y_bin = coerce_xy_binary(X, y)
        return float((y_pred == y_bin.astype(np.int8)).mean())
