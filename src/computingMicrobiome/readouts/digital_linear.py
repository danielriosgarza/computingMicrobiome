"""Digital-evolution linear readout with discrete genomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .base import coerce_xy_binary


@dataclass
class _Genome:
    """Discrete genotype for a sparse linear readout."""

    feature_idx: np.ndarray  # shape (n_genes,), int
    weight_code: np.ndarray  # shape (n_genes,), int in [0, n_codes)
    bias_code: int

    @property
    def n_genes(self) -> int:
        return int(self.feature_idx.size)


class DigitalLinearReadout:
    """Linear readout trained by digital evolution (steady-state).

    The genotype is a list of discrete genes mapping feature indices to
    quantized weight codes, plus a quantized bias code.
    """

    def __init__(
        self,
        *,
        population_size: int = 128,
        generations: int = 3000,
        tournament_size: int = 5,
        death_tournament_size: int = 5,
        min_genes: int = 4,
        max_genes: int = 64,
        init_genes: int | None = 24,
        weight_values: Sequence[float] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        weight_scale: float = 0.25,
        mutation_rate_feature: float = 0.25,
        mutation_rate_weight: float = 0.25,
        mutation_rate_bias: float = 0.1,
        mutation_rate_insert: float = 0.1,
        mutation_rate_delete: float = 0.1,
        mutation_rate_swap: float = 0.05,
        complexity_penalty: float = 0.0,
        batch_size: int | None = None,
        accept_only_improving: bool = True,
        normalize_features: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_size = int(tournament_size)
        self.death_tournament_size = int(death_tournament_size)
        self.min_genes = int(min_genes)
        self.max_genes = int(max_genes)
        self.init_genes = None if init_genes is None else int(init_genes)
        self.weight_values = tuple(float(v) for v in weight_values)
        self.weight_scale = float(weight_scale)
        self.mutation_rate_feature = float(mutation_rate_feature)
        self.mutation_rate_weight = float(mutation_rate_weight)
        self.mutation_rate_bias = float(mutation_rate_bias)
        self.mutation_rate_insert = float(mutation_rate_insert)
        self.mutation_rate_delete = float(mutation_rate_delete)
        self.mutation_rate_swap = float(mutation_rate_swap)
        self.complexity_penalty = float(complexity_penalty)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.accept_only_improving = bool(accept_only_improving)
        self.normalize_features = bool(normalize_features)
        self._rng = rng

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.best_genome_: dict | None = None
        self.fitness_history_: list[float] = []
        self.fitness_mean_history_: list[float] = []
        self.genome_size_history_: list[float] = []
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

    def _n_codes(self) -> int:
        return len(self.weight_values)

    def _decode_code(self, code: int) -> float:
        values = np.asarray(self.weight_values, dtype=float)
        clipped = int(np.clip(code, 0, values.size - 1))
        return float(values[clipped] * self.weight_scale)

    def _random_gene_count(self, rng: np.random.Generator) -> int:
        if self.init_genes is not None:
            return int(np.clip(self.init_genes, self.min_genes, self.max_genes))
        return int(rng.integers(self.min_genes, self.max_genes + 1))

    def _random_genome(self, n_features: int, rng: np.random.Generator) -> _Genome:
        n_genes = self._random_gene_count(rng)
        feature_idx = rng.integers(0, n_features, size=n_genes, dtype=np.int64)
        weight_code = rng.integers(0, self._n_codes(), size=n_genes, dtype=np.int64)
        bias_code = int(rng.integers(0, self._n_codes()))
        return _Genome(feature_idx=feature_idx, weight_code=weight_code, bias_code=bias_code)

    @staticmethod
    def _copy_genome(g: _Genome) -> _Genome:
        return _Genome(
            feature_idx=g.feature_idx.copy(),
            weight_code=g.weight_code.copy(),
            bias_code=int(g.bias_code),
        )

    def _mutate_code_local(self, code: int, rng: np.random.Generator) -> int:
        step = int(rng.choice((-1, 1)))
        return int(np.clip(code + step, 0, self._n_codes() - 1))

    def _mutate_genome(
        self, g: _Genome, n_features: int, rng: np.random.Generator
    ) -> _Genome:
        child = self._copy_genome(g)
        changed = False

        if child.n_genes > 0 and rng.random() < self.mutation_rate_feature:
            i = int(rng.integers(0, child.n_genes))
            child.feature_idx[i] = int(rng.integers(0, n_features))
            changed = True

        if child.n_genes > 0 and rng.random() < self.mutation_rate_weight:
            i = int(rng.integers(0, child.n_genes))
            child.weight_code[i] = self._mutate_code_local(int(child.weight_code[i]), rng)
            changed = True

        if rng.random() < self.mutation_rate_bias:
            child.bias_code = self._mutate_code_local(child.bias_code, rng)
            changed = True

        if child.n_genes < self.max_genes and rng.random() < self.mutation_rate_insert:
            ins_pos = int(rng.integers(0, child.n_genes + 1))
            ins_feat = int(rng.integers(0, n_features))
            ins_code = int(rng.integers(0, self._n_codes()))
            child.feature_idx = np.insert(child.feature_idx, ins_pos, ins_feat)
            child.weight_code = np.insert(child.weight_code, ins_pos, ins_code)
            changed = True

        if child.n_genes > self.min_genes and rng.random() < self.mutation_rate_delete:
            del_pos = int(rng.integers(0, child.n_genes))
            child.feature_idx = np.delete(child.feature_idx, del_pos)
            child.weight_code = np.delete(child.weight_code, del_pos)
            changed = True

        if child.n_genes >= 2 and rng.random() < self.mutation_rate_swap:
            i = int(rng.integers(0, child.n_genes))
            j = int(rng.integers(0, child.n_genes))
            if i != j:
                child.feature_idx[i], child.feature_idx[j] = (
                    int(child.feature_idx[j]),
                    int(child.feature_idx[i]),
                )
                child.weight_code[i], child.weight_code[j] = (
                    int(child.weight_code[j]),
                    int(child.weight_code[i]),
                )
                changed = True

        if not changed:
            if child.n_genes > 0:
                i = int(rng.integers(0, child.n_genes))
                child.weight_code[i] = self._mutate_code_local(int(child.weight_code[i]), rng)
            else:
                child.feature_idx = np.array([int(rng.integers(0, n_features))], dtype=np.int64)
                child.weight_code = np.array([int(rng.integers(0, self._n_codes()))], dtype=np.int64)
            changed = True

        return child

    def _genome_to_dense(self, genome: _Genome, n_features: int) -> tuple[np.ndarray, float]:
        coef = np.zeros(n_features, dtype=float)
        for i in range(genome.n_genes):
            idx = int(genome.feature_idx[i])
            coef[idx] += self._decode_code(int(genome.weight_code[i]))
        bias = self._decode_code(genome.bias_code)
        return coef, float(bias)

    def _logits(self, X: np.ndarray, genome: _Genome) -> np.ndarray:
        logits = np.full(X.shape[0], self._decode_code(genome.bias_code), dtype=float)
        for i in range(genome.n_genes):
            idx = int(genome.feature_idx[i])
            w = self._decode_code(int(genome.weight_code[i]))
            logits += w * X[:, idx]
        return logits

    def _fitness(self, X: np.ndarray, y: np.ndarray, genome: _Genome) -> float:
        logits = self._logits(X, genome)
        loss = np.logaddexp(0.0, logits) - y * logits
        return float(-loss.mean() - self.complexity_penalty * genome.n_genes)

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

    def fit(
        self, X: np.ndarray, y: Iterable[int], rng: np.random.Generator | None = None
    ):
        if self.min_genes < 0:
            raise ValueError("min_genes must be >= 0")
        if self.max_genes < self.min_genes:
            raise ValueError("max_genes must be >= min_genes")
        if self._n_codes() < 2:
            raise ValueError("weight_values must contain at least two values")

        X_arr, y_arr = coerce_xy_binary(X, y)
        if self.normalize_features:
            self._feature_mean = X_arr.mean(axis=0, keepdims=True)
            std = X_arr.std(axis=0, keepdims=True)
            self._feature_std = np.where(std > 0, std, 1.0)
            X_arr = (X_arr - self._feature_mean) / self._feature_std

        rng = self._get_rng(rng)
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        pop_size = max(2, self.population_size)
        population = [self._random_genome(n_features, rng) for _ in range(pop_size)]
        replacements = 0

        self.fitness_history_ = []
        self.fitness_mean_history_ = []
        self.genome_size_history_ = []
        self.replacement_rate_history_ = []

        for gen_idx in range(self.generations):
            X_batch, y_batch = self._sample_batch(X_arr, y_arr, rng)
            fitness = np.array(
                [self._fitness(X_batch, y_batch, g) for g in population], dtype=float
            )

            best_idx = int(np.argmax(fitness))
            self.fitness_history_.append(float(fitness[best_idx]))
            self.fitness_mean_history_.append(float(fitness.mean()))
            self.genome_size_history_.append(float(population[best_idx].n_genes))

            parent_idx = self._select_parent_index(fitness, rng)
            death_idx = self._select_death_index(fitness, rng)

            child = self._mutate_genome(population[parent_idx], n_features, rng)
            child_fit = self._fitness(X_batch, y_batch, child)

            should_replace = True
            if self.accept_only_improving:
                should_replace = child_fit > float(fitness[death_idx])

            if should_replace:
                population[death_idx] = child
                replacements += 1

            self.replacement_rate_history_.append(replacements / float(gen_idx + 1))

        fitness_full = np.array(
            [self._fitness(X_arr, y_arr, g) for g in population], dtype=float
        )
        best_idx = int(np.argmax(fitness_full))
        best = population[best_idx]

        self.coef_, self.intercept_ = self._genome_to_dense(best, n_features)
        self.best_genome_ = {
            "feature_idx": best.feature_idx.astype(int).tolist(),
            "weight_code": best.weight_code.astype(int).tolist(),
            "bias_code": int(best.bias_code),
            "n_genes": int(best.n_genes),
        }
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
