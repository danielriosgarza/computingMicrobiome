"""Generate a PDF with 256 occupancy heatmaps (one per 8-bit pattern).

Uses the same IBM pulse config and task schedule as task_4_learn_8_bit_with_ibm_pulse.
Each page shows the final species occupancy grid after injecting one of the 256 patterns.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from computingMicrobiome.benchmarks.episode_runner import run_reservoir_episode
from computingMicrobiome.benchmarks.k_bit_memory_bm import create_input_streams
from computingMicrobiome.ibm import make_ibm_config_from_species
from computingMicrobiome.reservoirs.factory import make_reservoir
from computingMicrobiome.utils import create_input_locations, int_to_bits

BITS = 8
RECURRENCE = 4
ITR = 12
D_PERIOD = 8
SEED_TRAIN = 0

OUT_DIR = pathlib.Path(__file__).resolve().parent / "task_4_pulse_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# Same config as in task_4_learn_8_bit_with_ibm_pulse.py
IBM_PULSE_CFG = make_ibm_config_from_species(
    species_indices=[0, 1, 17, 20, 21, 40, 41],
    height=8,
    width_grid=8,
    overrides={
        "basal_energy": 4,
        "dilution_p": 0.05,
        "basal_init": True,
        "pulse_radius": 2,
        "pulse_toxin_conc": 180,
        "pulse_popular_conc": 200,
        "left_source_outcompete_margin": 1,
        "left_source_colonize_empty": True,
    },
)


def main() -> None:
    width = int(IBM_PULSE_CFG["height"]) * int(IBM_PULSE_CFG["width_grid"])
    rng = np.random.default_rng(SEED_TRAIN)
    input_locations = create_input_locations(width, RECURRENCE, 4, rng)
    reservoir = make_reservoir(reservoir_kind="ibm_pulse", reservoir_config=IBM_PULSE_CFG)
    n_species = reservoir.n_species

    # Colormap: one color for empty (-1) and one per species (0..n_species-1)
    cmap_colors = ["#f0f0f0"]  # empty
    cmap_colors.extend(plt.cm.tab10(np.linspace(0, 1, max(n_species, 1)))[:n_species])
    cmap = mcolors.ListedColormap(cmap_colors)
    # Plot values: -1 -> 0, 0..n_species-1 -> 1..n_species
    vmin, vmax = 0, n_species

    pdf_path = OUT_DIR / "task_4_pulse_256_occupancy.pdf"
    with plt.rc_context({"figure.max_open_warning": 0}):
        with PdfPages(pdf_path) as pdf:
            for val in range(256):
                bits_arr = int_to_bits(val, BITS)
                input_streams = create_input_streams(bits_arr, D_PERIOD)
                run_reservoir_episode(
                    input_streams=input_streams,
                    reservoir=reservoir,
                    itr=ITR,
                    input_locations=input_locations,
                    rng=rng,
                    reg=None,
                    collect_states=False,
                    x0_mode="zeros",
                )
                occ = reservoir.get_occupancy()
                # Map -1,0,1,... to 0,1,2,... for colormap
                plot_arr = (occ.astype(np.int32) + 1).clip(0, n_species)

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(plot_arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xlabel("column")
                ax.set_ylabel("row")
                ax.set_xticks(range(occ.shape[1]))
                ax.set_yticks(range(occ.shape[0]))
                bits_str = ",".join(str(b) for b in bits_arr.tolist())
                ax.set_title(f"Species occupancy (task 4 pulse)\nPattern {val}: ({bits_str})")
                fig.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

    print(f"Saved {pdf_path} (256 pages).")


if __name__ == "__main__":
    main()
