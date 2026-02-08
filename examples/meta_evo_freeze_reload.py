"""Train a meta-evolutionary readout, then freeze/save/reload it.

Run:
    python examples/meta_evo_freeze_reload.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from computingMicrobiome.readouts.factory import make_readout
from computingMicrobiome.readouts.meta_evolutionary_linear import FrozenLinearReadout


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 6))
    w = np.array([1.5, -2.0, 0.7, 0.0, 0.3, -0.4], dtype=float)
    y = ((X @ w) > 0.0).astype(np.int8)

    reg = make_readout(
        "meta_evo",
        config={
            "population_size": 20,
            "generations": 30,
            "tournament_size": 4,
            "elite_count": 2,
            "min_adaptation_steps": 2,
            "max_adaptation_steps": 12,
            "normalize_features": True,
        },
        rng=np.random.default_rng(1),
    )
    reg.fit(X, y)

    artifact_path = Path("artifacts") / "meta_evo_frozen_readout.json"
    frozen = reg.freeze()
    frozen.save_json(artifact_path)
    loaded = FrozenLinearReadout.load_json(artifact_path)

    acc_train = reg.score(X, y)
    preds_match = np.array_equal(frozen.predict(X), loaded.predict(X))

    print(f"Trained meta_evo readout accuracy: {acc_train:.4f}")
    print(f"Saved artifact: {artifact_path}")
    print(f"Reloaded predictions identical: {preds_match}")


if __name__ == "__main__":
    main()
