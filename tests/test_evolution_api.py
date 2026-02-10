from __future__ import annotations

from computingMicrobiome.evolution import run_evolution_of_learners


def test_run_evolution_of_learners_smoke() -> None:
    res = run_evolution_of_learners(
        task="memory",
        representation="direct",
        learner="perceptron",
        engine="moran",
        n_generations=2,
        population_size=4,
        support_size=4,
        challenge_size=4,
        seed=0,
        task_params={"bits": 3},
    )
    assert res.history
    assert len(res.history) == 2


def test_comparative_smoke_direct_vs_reservoir() -> None:
    # Direct representation
    res_direct = run_evolution_of_learners(
        task="memory",
        representation="direct",
        learner="perceptron",
        engine="moran",
        n_generations=1,
        population_size=2,
        support_size=2,
        challenge_size=2,
        seed=1,
        task_params={"bits": 2},
    )
    # Reservoir representation
    res_reservoir = run_evolution_of_learners(
        task="memory",
        representation="reservoir",
        learner="perceptron",
        engine="moran",
        n_generations=1,
        population_size=2,
        support_size=2,
        challenge_size=2,
        seed=1,
        task_params={"bits": 2},
        representation_params={
            "rule_number": 110,
            "width": 50,
            "boundary": "periodic",
            "recurrence": 2,
            "itr": 1,
            "d_period": 10,
        },
    )

    assert res_direct.history and res_reservoir.history


def test_run_with_print_progress_smoke() -> None:
    res = run_evolution_of_learners(
        task="memory",
        representation="direct",
        learner="perceptron",
        engine="moran",
        n_generations=2,
        population_size=4,
        support_size=4,
        challenge_size=4,
        seed=0,
        task_params={"bits": 3},
        progress=True,
        progress_mode="print",
        progress_every=1,
    )
    assert len(res.history) == 2

