"""Generate a PDF with 256 occupancy heatmaps (one per 8-bit pattern).

Uses the exact Task-4 pulse-matched IBM setup from
`task_4_learn_8_bit_with_ibm_pulse_matched.py`.
Each page shows side-by-side occupancy snapshots at the output-window ticks
that are used for prediction.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from computingMicrobiome.benchmarks.k_bit_memory_bm import create_input_streams
from computingMicrobiome.ibm import (
    make_channel_to_resource_from_config,
    make_ibm_config_from_species,
)
from computingMicrobiome.reservoirs.factory import make_reservoir
from computingMicrobiome.utils import create_input_locations, int_to_bits

BITS = 8
BOUNDARY = "periodic"
RECURRENCE = 4
ITR = 12
D_PERIOD = 8
SEED_TRAIN = 0

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_4_pulse_matched_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# Delay in simulation steps from bit injection to output window.
TRACE_DEPTH = (D_PERIOD + BITS) * (ITR + 1) + 8
N_CHANNELS = 4

# IBM reservoir dynamics (exactly as task_4_learn_8_bit_with_ibm_pulse_matched.py)
IBM_DIFF_NUMER = 1
IBM_DILUTION_P = 0.5
IBM_INJECT_SCALE = 2.0
PULSE_RADIUS = 0
PULSE_TOXIN_CONC = 180
PULSE_POPULAR_CONC = 200

IBM_PULSE_CFG = make_ibm_config_from_species(
    species_indices=[0, 1, 20, 21, 40, 41],
    height=16,
    width_grid=16,
    overrides={
        "state_width_mode": "raw",
        "input_trace_depth": TRACE_DEPTH,
        "input_trace_channels": N_CHANNELS,
        "input_trace_decay": 1.0,
        "inject_scale": IBM_INJECT_SCALE,
        "dilution_p": IBM_DILUTION_P,
        "diff_numer": IBM_DIFF_NUMER,
        "inject_mode": "pulse_bit",
        "pulse_radius": PULSE_RADIUS,
        "pulse_toxin_conc": PULSE_TOXIN_CONC,
        "pulse_popular_conc": PULSE_POPULAR_CONC,
        "basal_init": False,
        "left_source_enabled": True,
        "left_source_species": [-1] * 5 + [0, 1, 2, 3, 4, 5] + [-1] * 5,
        "left_source_colonize_empty": True,
        "left_source_outcompete_margin": 1,
    },
)
IBM_PULSE_CFG["channel_to_resource"] = make_channel_to_resource_from_config(
    IBM_PULSE_CFG, N_CHANNELS
)


def capture_prediction_tick_occupancies(
    *,
    input_streams: np.ndarray,
    reservoir,
    itr: int,
    input_locations: np.ndarray,
    rng: np.random.Generator,
    bits: int,
) -> tuple[list[np.ndarray], int]:
    """Capture occupancy at the output-window ticks used for prediction."""
    L = int(input_streams.shape[0])
    iter_between = int(itr) + 1
    T = L * iter_between
    output_start = L - int(bits)
    if output_start < 0:
        raise ValueError("bits cannot exceed number of ticks in episode")

    reservoir.reset(rng, x0_mode="zeros")
    channel_idx = np.arange(input_locations.size) % input_streams.shape[1]
    tick = 0
    occ_frames: list[np.ndarray] = []

    for i in range(T):
        if i % iter_between == 0:
            in_bits = input_streams[tick]
            reservoir.inject(in_bits, input_locations, channel_idx)
            if tick >= output_start:
                occ_frames.append(reservoir.get_occupancy().copy())
            tick += 1
        reservoir.step(rng)

    if len(occ_frames) != int(bits):
        raise RuntimeError("failed to capture all output-window occupancy frames")
    return occ_frames, output_start


def main() -> None:
    width = int(IBM_PULSE_CFG["height"]) * int(IBM_PULSE_CFG["width_grid"])
    rng = np.random.default_rng(SEED_TRAIN)
    input_locations = create_input_locations(width, RECURRENCE, N_CHANNELS, rng)
    n_species = int(IBM_PULSE_CFG["n_species"])

    # Colormap: one color for empty (-1) and one per species (0..n_species-1)
    cmap_colors = ["#f0f0f0"]  # empty
    cmap_colors.extend(plt.cm.tab10(np.linspace(0, 1, max(n_species, 1)))[:n_species])
    cmap = mcolors.ListedColormap(cmap_colors)
    # Plot values: -1 -> 0, 0..n_species-1 -> 1..n_species
    vmin, vmax = 0, n_species

    pdf_path = OUT_DIR / "task_4_pulse_256_prediction_tick_occupancy.pdf"
    with plt.rc_context({"figure.max_open_warning": 0}):
        with PdfPages(pdf_path) as pdf:
            for val in range(256):
                bits_arr = int_to_bits(val, BITS)
                input_streams = create_input_streams(bits_arr, D_PERIOD)
                reservoir = make_reservoir(
                    reservoir_kind="ibm_pulse",
                    rule_number=110,
                    width=width,
                    boundary=BOUNDARY,
                    reservoir_config=IBM_PULSE_CFG,
                )
                occ_frames, output_start = capture_prediction_tick_occupancies(
                    input_streams=input_streams,
                    reservoir=reservoir,
                    itr=ITR,
                    input_locations=input_locations,
                    rng=rng,
                    bits=BITS,
                )
                fig, axes = plt.subplots(1, BITS, figsize=(2.5 * BITS, 3.5))
                if BITS == 1:
                    axes = [axes]
                for j, occ in enumerate(occ_frames):
                    ax = axes[j]
                    # Map -1,0,1,... to 0,1,2,... for colormap
                    plot_arr = (occ.astype(np.int32) + 1).clip(0, n_species)
                    ax.imshow(
                        plot_arr,
                        aspect="auto",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                    ax.set_title(f"tick {output_start + j}", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])

                axes[0].set_ylabel("row")
                axes[0].set_xlabel("column")
                bits_str = ",".join(str(b) for b in bits_arr.tolist())
                fig.suptitle(
                    "Output-window occupancies used for prediction "
                    f"(task 4 pulse matched)\nPattern {val}: ({bits_str})",
                    fontsize=11,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.88])
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

    print(f"Saved {pdf_path} (256 pages).")


if __name__ == "__main__":
    main()
