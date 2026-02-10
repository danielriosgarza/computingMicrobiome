"""Selection operators for evolution-of-learners."""

from __future__ import annotations

import numpy as np


def tournament_select_parent(
    fitness: np.ndarray, t_size: int, rng: np.random.Generator
) -> int:
    """Select an index via tournament selection favouring higher fitness."""
    if t_size < 2:
        raise ValueError("t_size must be >= 2")
    if fitness.size < t_size:
        raise ValueError("fitness size must be >= t_size")
    idx = rng.choice(fitness.size, size=t_size, replace=False)
    best_local = idx[np.argmax(fitness[idx])]
    return int(best_local)


def tournament_select_death(
    fitness: np.ndarray, t_size: int, rng: np.random.Generator
) -> int:
    """Select an index via tournament selection favouring lower fitness."""
    if t_size < 2:
        raise ValueError("t_size must be >= 2")
    if fitness.size < t_size:
        raise ValueError("fitness size must be >= t_size")
    idx = rng.choice(fitness.size, size=t_size, replace=False)
    worst_local = idx[np.argmin(fitness[idx])]
    return int(worst_local)


__all__ = ["tournament_select_parent", "tournament_select_death"]

