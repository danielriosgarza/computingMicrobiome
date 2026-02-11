"""CRM-only population kinetics simulation and plots.

This experiment runs CRM dynamics without any task/readout evolution and
visualizes species/resources kinetics through time.

Run:
    python -m experiments.crm_population_kinetics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from computingMicrobiome.reservoirs.crm_backend import CRMReservoirBackend

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _decode_state(
    state: np.ndarray, *, n_species: int, n_resources: int, height: int, width_grid: int
) -> tuple[np.ndarray, np.ndarray]:
    cells = height * width_grid
    s_end = n_species * cells
    x_flat = state[:s_end]
    r_flat = state[s_end : s_end + (n_resources * cells)]
    X = x_flat.reshape(n_species, height, width_grid)
    R = r_flat.reshape(n_resources, height, width_grid)
    return X, R


def _random_crm_config(
    rng: np.random.Generator,
    *,
    n_species: int,
    n_resources: int,
    height: int,
    width_grid: int,
) -> dict:
    reaction = rng.uniform(0.05, 0.40, size=(n_species, n_resources)).astype(np.float32)
    consumption = rng.uniform(0.05, 0.30, size=(n_species, n_resources)).astype(np.float32)
    inflow = rng.uniform(0.05, 0.35, size=(n_resources,)).astype(np.float32)
    diff_species = rng.uniform(0.005, 0.05, size=(n_species,)).astype(np.float32)
    diff_resources = rng.uniform(0.01, 0.08, size=(n_resources,)).astype(np.float32)

    return {
        "height": height,
        "width_grid": width_grid,
        "n_species": n_species,
        "n_resources": n_resources,
        "reaction_matrix": reaction.tolist(),
        "consumption_matrix": consumption.tolist(),
        "resource_inflow": inflow.tolist(),
        "diffusion_species": diff_species.tolist(),
        "diffusion_resources": diff_resources.tolist(),
        "dt": float(rng.uniform(0.02, 0.08)),
        "dilution": float(rng.uniform(0.002, 0.03)),
        "noise_std": float(rng.uniform(0.0, 0.01)),
        # Keep identity projection so we can decode species/resource fields directly.
        "projection": {"kind": "identity"},
    }


def _species_diversity(species_totals: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = species_totals / np.maximum(species_totals.sum(axis=1, keepdims=True), eps)
    return -np.sum(p * np.log2(np.maximum(p, eps)), axis=1)


def _plot_time_series(
    species_totals: np.ndarray,
    resource_totals: np.ndarray,
    diversity: np.ndarray,
    out_dir: Path,
) -> None:
    t = np.arange(species_totals.shape[0])

    plt.figure(figsize=(10, 4.5))
    for i in range(species_totals.shape[1]):
        plt.plot(t, species_totals[:, i], label=f"species_{i}")
    plt.xlabel("time step")
    plt.ylabel("total biomass")
    plt.title("CRM population kinetics: species totals")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "crm_species_kinetics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    for j in range(resource_totals.shape[1]):
        plt.plot(t, resource_totals[:, j], label=f"resource_{j}")
    plt.xlabel("time step")
    plt.ylabel("total resource mass")
    plt.title("CRM population kinetics: resource totals")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "crm_resource_kinetics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(t, species_totals.sum(axis=1), label="total_species_biomass")
    plt.plot(t, resource_totals.sum(axis=1), label="total_resource_mass")
    plt.plot(t, diversity, label="species_diversity_bits", linestyle="--")
    plt.xlabel("time step")
    plt.ylabel("aggregate value")
    plt.title("CRM aggregate kinetics")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "crm_aggregate_kinetics.png", dpi=180)
    plt.close()


def _plot_final_snapshots(X: np.ndarray, R: np.ndarray, out_dir: Path) -> None:
    n_species = X.shape[0]
    n_resources = R.shape[0]
    n_rows = max(n_species, n_resources)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 2.3 * n_rows), squeeze=False)

    for i in range(n_rows):
        ax_s = axes[i, 0]
        if i < n_species:
            im_s = ax_s.imshow(X[i], cmap="viridis", aspect="auto")
            ax_s.set_title(f"species_{i} final")
            fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        else:
            ax_s.axis("off")

        ax_r = axes[i, 1]
        if i < n_resources:
            im_r = ax_r.imshow(R[i], cmap="magma", aspect="auto")
            ax_r.set_title(f"resource_{i} final")
            fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
        else:
            ax_r.axis("off")

    for ax in axes.reshape(-1):
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(out_dir / "crm_final_spatial_fields.png", dpi=180)
    plt.close()


def run_crm_population_kinetics(
    *,
    steps: int,
    seed: int,
    n_species: int,
    n_resources: int,
    height: int,
    width_grid: int,
    x0_mode: str,
    randomize_config: bool,
    out_dir: Path,
) -> None:
    rng = np.random.default_rng(seed)

    if randomize_config:
        crm_config = _random_crm_config(
            rng,
            n_species=n_species,
            n_resources=n_resources,
            height=height,
            width_grid=width_grid,
        )
    else:
        crm_config = {
            "height": height,
            "width_grid": width_grid,
            "n_species": n_species,
            "n_resources": n_resources,
            "reaction_matrix": (np.ones((n_species, n_resources), dtype=np.float32) * 0.25).tolist(),
            "consumption_matrix": (np.ones((n_species, n_resources), dtype=np.float32) * 0.20).tolist(),
            "resource_inflow": (np.ones((n_resources,), dtype=np.float32) * 0.2).tolist(),
            "diffusion_species": (np.ones((n_species,), dtype=np.float32) * 0.02).tolist(),
            "diffusion_resources": (np.ones((n_resources,), dtype=np.float32) * 0.05).tolist(),
            "dt": 0.05,
            "dilution": 0.01,
            "noise_std": 0.0,
            "projection": {"kind": "identity"},
        }

    backend = CRMReservoirBackend(config=crm_config)
    backend.reset(rng, x0_mode=x0_mode)

    species_totals = np.zeros((steps, n_species), dtype=np.float64)
    resource_totals = np.zeros((steps, n_resources), dtype=np.float64)

    final_X = None
    final_R = None
    for t in range(steps):
        state = backend.get_state()
        X, R = _decode_state(
            state,
            n_species=n_species,
            n_resources=n_resources,
            height=height,
            width_grid=width_grid,
        )
        species_totals[t] = X.reshape(n_species, -1).sum(axis=1)
        resource_totals[t] = R.reshape(n_resources, -1).sum(axis=1)
        final_X, final_R = X, R
        backend.step(rng)

    assert final_X is not None and final_R is not None
    diversity = _species_diversity(species_totals)

    _plot_time_series(species_totals, resource_totals, diversity, out_dir)
    _plot_final_snapshots(final_X, final_R, out_dir)

    payload = {
        "config": {
            "steps": steps,
            "seed": seed,
            "n_species": n_species,
            "n_resources": n_resources,
            "height": height,
            "width_grid": width_grid,
            "x0_mode": x0_mode,
            "randomize_config": randomize_config,
            "crm_config": crm_config,
        },
        "timeseries": {
            "species_totals": species_totals.tolist(),
            "resource_totals": resource_totals.tolist(),
            "species_diversity_bits": diversity.tolist(),
        },
        "final": {
            "species_totals": species_totals[-1].tolist(),
            "resource_totals": resource_totals[-1].tolist(),
            "total_species_biomass": float(species_totals[-1].sum()),
            "total_resource_mass": float(resource_totals[-1].sum()),
            "final_diversity_bits": float(diversity[-1]),
        },
    }
    out_path = out_dir / "crm_population_kinetics_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Saved plots in:", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRM-only population kinetics simulation")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_species", type=int, default=4)
    parser.add_argument("--n_resources", type=int, default=3)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--width_grid", type=int, default=12)
    parser.add_argument("--x0_mode", type=str, default="random", choices=["zeros", "random"])
    parser.add_argument(
        "--randomize-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample random asymmetric CRM kinetics parameters "
            "(default: enabled; pass --no-randomize-config for symmetric defaults)"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/crm_population_kinetics",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_crm_population_kinetics(
        steps=args.steps,
        seed=args.seed,
        n_species=args.n_species,
        n_resources=args.n_resources,
        height=args.height,
        width_grid=args.width_grid,
        x0_mode=args.x0_mode,
        randomize_config=bool(args.randomize_config),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
