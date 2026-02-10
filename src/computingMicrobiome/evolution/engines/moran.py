"""Moran-style evolution engine for evolution-of-learners experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

from ..base import (
    GenerationMetrics,
    IndividualState,
    LearnerProtocol,
    RepresentationProtocol,
    TaskSamplerProtocol,
)
from ..config import EvolutionConfig
from ..mutation import mutate_perceptron_genotype
from ..results import EvolutionRunResult
from ..selection import tournament_select_death, tournament_select_parent


LearnerFactory = Callable[[], LearnerProtocol]


@dataclass
class MoranEvolutionEngine:
    """Outer-loop Moran-style evolutionary engine."""

    @staticmethod
    def _make_generation_iterator(
        generations: int,
        *,
        progress: bool,
        progress_mode: str,
        progress_desc: str,
    ) -> tuple[Iterable[int], object | None, bool]:
        """Build generation iterator with optional progress UI.

        Returns:
            (iterator, progress_bar_object, use_print_progress)
        """
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

    def run(
        self,
        task_sampler: TaskSamplerProtocol,
        representation: RepresentationProtocol,
        learner_factory: LearnerFactory,
        evolution_config: EvolutionConfig,
        rng: np.random.Generator,
        initial_genotypes: List[object],
        *,
        progress: bool = False,
        progress_mode: str = "auto",
        progress_every: int = 10,
        progress_desc: str = "Evolution",
    ) -> EvolutionRunResult:
        """Execute a full evolutionary run.

        The engine enforces the following invariants:
        - Fixed population size across generations.
        - Same support/challenge sets for all individuals within a generation.
        - Deterministic behaviour given a fixed RNG seed.
        """
        pop_size = evolution_config.population_size
        if len(initial_genotypes) != pop_size:
            raise ValueError("initial_genotypes length must equal population_size")
        if progress_every < 1:
            raise ValueError("progress_every must be >= 1")

        individuals: List[IndividualState] = [
            IndividualState(id=i, genotype=initial_genotypes[i]) for i in range(pop_size)
        ]
        history: List[GenerationMetrics] = []

        iterator, progress_bar, use_print_progress = self._make_generation_iterator(
            evolution_config.generations,
            progress=progress,
            progress_mode=progress_mode,
            progress_desc=progress_desc,
        )

        for g in iterator:
            # 1) Draw shared support/challenge sets.
            raw_sup_x, _ = task_sampler.sample_support(rng, evolution_config.support_size)
            raw_ch_x, _ = task_sampler.sample_challenge(
                rng, evolution_config.challenge_size
            )
            X_sup, y_sup = representation.transform(raw_sup_x, rng)
            X_ch, y_ch = representation.transform(raw_ch_x, rng)

            # 2) Evaluate population.
            fitness_vals = np.empty(pop_size, dtype=float)
            adaptation_gains = np.zeros(pop_size, dtype=float)

            for i, ind in enumerate(individuals):
                learner = learner_factory()
                # Pre-train fitness (optional).
                learner.reset_from_genotype(ind.genotype, rng)
                pre_fit = learner.score(X_ch, y_ch)
                ind.pre_train_fitness = pre_fit

                # Train on support set.
                learner.fit(X_sup, y_sup)
                post_fit = learner.score(X_ch, y_ch)
                ind.post_train_fitness = post_fit
                ind.fitness = post_fit

                fitness_vals[i] = post_fit
                adaptation_gains[i] = post_fit - pre_fit

            # 3) Replacement events within this generation.
            replacements = 0
            for _ in range(evolution_config.birth_events_per_generation):
                parent_idx = tournament_select_parent(
                    fitness_vals, evolution_config.parent_tournament_size, rng
                )
                parent = individuals[parent_idx]

                child_genotype = mutate_perceptron_genotype(parent.genotype, evolution_config, rng)

                # Evaluate child using the SAME batches for fairness.
                learner = learner_factory()
                learner.reset_from_genotype(child_genotype, rng)
                learner.fit(X_sup, y_sup)
                child_fitness = learner.score(X_ch, y_ch)

                death_idx = tournament_select_death(
                    fitness_vals, evolution_config.death_tournament_size, rng
                )

                if (not evolution_config.accept_only_improving) or (
                    child_fitness > fitness_vals[death_idx]
                ):
                    # Replace individual at death_idx with new child.
                    individuals[death_idx] = IndividualState(
                        id=individuals[death_idx].id,
                        genotype=child_genotype,
                        fitness=child_fitness,
                        pre_train_fitness=None,
                        post_train_fitness=child_fitness,
                    )
                    fitness_vals[death_idx] = child_fitness
                    replacements += 1

            # 4) Record generation metrics.
            mean_fit = float(fitness_vals.mean())
            best_fit = float(fitness_vals.max())
            std_fit = float(fitness_vals.std())
            mean_gain = float(adaptation_gains.mean())
            replacement_rate = float(replacements) / float(
                evolution_config.birth_events_per_generation
            )

            history.append(
                GenerationMetrics(
                    generation=g,
                    mean_fitness=mean_fit,
                    best_fitness=best_fit,
                    std_fitness=std_fit,
                    mean_adaptation_gain=mean_gain,
                    replacement_rate=replacement_rate,
                )
            )

            if progress_bar is not None:
                progress_bar.set_postfix(
                    mean=f"{mean_fit:.3f}",
                    best=f"{best_fit:.3f}",
                    repl=f"{replacement_rate:.2f}",
                )
            elif use_print_progress and (
                (g + 1) % progress_every == 0
                or g == 0
                or g == evolution_config.generations - 1
            ):
                print(
                    f"[evolution] gen {g + 1}/{evolution_config.generations} "
                    f"mean={mean_fit:.4f} best={best_fit:.4f} repl={replacement_rate:.3f}"
                )

        if progress_bar is not None:
            progress_bar.close()

        best_idx = int(np.argmax([ind.fitness for ind in individuals]))
        best_genotype = individuals[best_idx].genotype

        result = EvolutionRunResult(
            config={
                "evolution": evolution_config.__dict__,
            },
            history=history,
            final_population_fitness=np.array(
                [ind.fitness if ind.fitness is not None else np.nan for ind in individuals],
                dtype=float,
            ),
            best_genotype=getattr(best_genotype, "__dict__", dict(best_genotype=best_genotype)),
            seed=evolution_config.seed,
        )
        return result


__all__ = ["MoranEvolutionEngine"]

