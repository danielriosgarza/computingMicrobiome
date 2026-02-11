"""Result dataclasses and helpers for evolution-of-learners experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base import GenerationMetrics


@dataclass
class EvolutionRunResult:
    """Container for the outcome of an evolutionary run."""

    config: Dict[str, Any]
    history: List[GenerationMetrics]
    final_population_fitness: np.ndarray
    best_genotype: Dict[str, Any]
    seed: int
    final_population_genotypes: List[Dict[str, Any]] = field(default_factory=list)
    population_composition_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a JSON-serialisable dictionary."""
        return {
            "config": self.config,
            "history": [asdict(m) for m in self.history],
            "final_population_fitness": self.final_population_fitness.tolist(),
            "best_genotype": self.best_genotype,
            "seed": self.seed,
            "final_population_genotypes": self.final_population_genotypes,
            "population_composition_history": self.population_composition_history,
        }

    def save_json(self, path: str | Path) -> None:
        """Save the result to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


__all__ = [
    "EvolutionRunResult",
]

