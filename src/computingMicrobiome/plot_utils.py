from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.decomposition import PCA

from .experiments.toy_addition_experiment import (
    build_reservoir_dataset,
    enumerate_addition_dataset,
    evaluate_linear_task,
)


DEFAULT_BG_COLOR = "#f7e9f0"  # very light pink
DEFAULT_ACTIVE_COLOR = "#21b0ff"  # neon blue
DEFAULT_BORDER_COLOR = "#ffffff"


def plot_spacetime(
    states: np.ndarray,
    title: str = "",
    *,
    cmap: Optional[str] = None,
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.2,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a space-time diagram for cellular automata states."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor(background_color)
    _plot_states(
        ax,
        states,
        cmap=cmap,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax.set_xlabel("cell index")
    ax.set_ylabel("time step")
    ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_spacetime_with_density(
    states: np.ndarray,
    title: str = "",
    *,
    cmap: Optional[str] = None,
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.2,
    show: bool = True,
) -> Tuple[plt.Axes, plt.Axes]:
    """Plot space-time plus density-over-time."""
    fig, (ax_img, ax_den) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax_img.set_facecolor(background_color)
    _plot_states(
        ax_img,
        states,
        cmap=cmap,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_img.set_ylabel("time step")
    ax_img.set_title(title)

    density = np.mean(states, axis=1)
    ax_den.set_facecolor(background_color)
    ax_den.plot(density, color=active_color)
    ax_den.set_ylabel("density")
    ax_den.set_xlabel("time step")
    ax_den.set_ylim(0.0, 1.0)

    if show:
        plt.show()
    return ax_img, ax_den


def plot_activity(
    states: np.ndarray,
    title: str = "Activity over time",
    *,
    line_color: str = DEFAULT_ACTIVE_COLOR,
    background_color: str = DEFAULT_BG_COLOR,
    show: bool = True,
) -> plt.Axes:
    """Plot the fraction of active cells at each time step."""
    density = np.mean(states, axis=1)
    _, ax = plt.subplots(figsize=(8, 3))
    ax.set_facecolor(background_color)
    ax.plot(density, color=line_color)
    ax.set_ylabel("density")
    ax.set_xlabel("time step")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_entropy(
    states: np.ndarray,
    title: str = "Shannon entropy over time",
    *,
    line_color: str = DEFAULT_ACTIVE_COLOR,
    background_color: str = DEFAULT_BG_COLOR,
    show: bool = True,
) -> plt.Axes:
    """Plot per-time-step Shannon entropy for binary CA states."""
    density = np.mean(states, axis=1)
    eps = 1e-12
    entropy = -(
        density * np.log2(density + eps)
        + (1 - density) * np.log2(1 - density + eps)
    )
    _, ax = plt.subplots(figsize=(8, 3))
    ax.set_facecolor(background_color)
    ax.plot(entropy, color=line_color)
    ax.set_ylabel("entropy (bits)")
    ax.set_xlabel("time step")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_rule_table(
    rule: int,
    *,
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.4,
    show: bool = True,
) -> plt.Axes:
    """Plot the 8 neighborhood patterns and outputs for an ECA rule."""
    if rule < 0 or rule > 255:
        raise ValueError("rule must be in [0, 255]")
    patterns = np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    outputs = np.array([(rule >> i) & 1 for i in range(7, -1, -1)])

    fig, ax = plt.subplots(figsize=(8, 2))
    grid = np.hstack([patterns, outputs[:, None]])
    ax.set_facecolor(background_color)
    _plot_states(
        ax,
        grid,
        cmap=None,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax.set_yticks(range(8))
    ax.set_yticklabels(["111", "110", "101", "100", "011", "010", "001", "000"])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["L", "C", "R", "Out"])
    ax.set_title(f"Rule {rule} table")
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("neighborhood / output")
    if show:
        plt.show()
    return ax


def plot_snapshot(
    states: np.ndarray,
    time_step: int,
    *,
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.2,
    show: bool = True,
) -> plt.Axes:
    """Plot a single time slice of the automaton."""
    if time_step < 0 or time_step >= states.shape[0]:
        raise IndexError("time_step out of range")
    _, ax = plt.subplots(figsize=(8, 1.5))
    ax.set_facecolor(background_color)
    _plot_states(
        ax,
        states[time_step : time_step + 1],
        cmap=None,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax.set_xlabel("cell index")
    ax.set_yticks([])
    ax.set_title(f"snapshot t={time_step}")
    if show:
        plt.show()
    return ax


def save_spacetime_image(
    states: np.ndarray,
    path: str | Path,
    *,
    title: str = "",
    cmap: Optional[str] = None,
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.2,
    dpi: int = 150,
) -> Path:
    """Save a space-time plot as a static image."""
    path = Path(path)
    if path.suffix.lower() != ".gif":
        if path.exists() and path.is_dir():
            path = path / "spacetime.gif"
        else:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "spacetime.gif"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ax = plot_spacetime(
        states,
        title,
        cmap=cmap,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
        show=False,
    )
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    return path


def make_spacetime_gif(
    states: np.ndarray,
    path: str | Path,
    *,
    reveal_mode: str = "grow",
    duration_ms: int = 60,
    cmap: Optional[str] = None,
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    cell_size: int = 6,
    border_px: int = 1,
    loop: int = 0,
) -> Path:
    """
    Create a GIF that reveals the space-time diagram over time.

    reveal_mode:
        - "grow": each frame shows rows [0:t]
        - "slide": a moving window of 50 rows
    """
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("imageio is required for GIF output") from exc

    path = Path(path)
    total_steps = states.shape[0]
    window = min(50, total_steps)
    target_height = total_steps if reveal_mode == "grow" else window
    frames = []
    for t in range(1, total_steps + 1):
        if reveal_mode == "grow":
            frame_states = states[:t]
        elif reveal_mode == "slide":
            start = max(0, t - window)
            frame_states = states[start:t]
        else:
            raise ValueError("reveal_mode must be 'grow' or 'slide'")
        frame_states = _pad_states(frame_states, target_height)
        frames.append(
            _states_to_rgb(
                frame_states,
                cmap=cmap,
                background_color=background_color,
                active_color=active_color,
                border_color=border_color,
                cell_size=cell_size,
                border_px=border_px,
            )
        )

    imageio.mimsave(str(path), frames, duration=duration_ms / 1000.0, loop=loop)
    return path


def plot_red_green_grid(correctness: np.ndarray, title: str = ""):
    """
    correctness: (n_trials, bits) with values {-1, +1}
    """
    correctness = np.asarray(correctness)
    n_trials, bits = correctness.shape

    cmap = ListedColormap(["#d62728", "#2ca02c"])  # red, green
    norm = BoundaryNorm([-1.5, 0.0, 1.5], cmap.N)

    # make tiles look like tiles
    cell = 0.30  # inches per cell
    fig_w = max(5.0, bits * cell + 2.0)
    fig_h = max(6.0, min(18.0, n_trials * cell + 2.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(correctness, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")

    ax.set_title(title)
    ax.set_xlabel("bit index (0..bits-1)")
    ax.set_ylabel("trial index")

    ax.set_xticks(np.arange(bits))
    ax.set_yticks(np.arange(0, n_trials, max(1, n_trials // 10)))

    # gridlines between cells
    ax.set_xticks(np.arange(-0.5, bits, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_trials, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.legend(
        handles=[
            mpatches.Patch(color="#2ca02c", label="correct"),
            mpatches.Patch(color="#d62728", label="wrong"),
        ],
        loc="upper right",
        framealpha=0.95,
    )

    plt.tight_layout()
    plt.show()

    acc = (correctness == 1).mean()
    print(f"Bit accuracy over all trials/bits: {acc:.3f}")


def plot_xor_episode(
    states: np.ndarray,
    inputs: np.ndarray,
    output_tick: int,
    y_true: int,
    y_pred: int | None = None,
    *,
    title: str = "",
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.2,
    show: bool = True,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot XOR episode with spacetime diagram and input timeline.

    inputs: shape (L, 4) where L is number of ticks
    output_tick: index of output tick in [0, L-1]
    """
    fig, (ax_states, ax_inputs) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )
    ax_states.set_facecolor(background_color)
    _plot_states(
        ax_states,
        states,
        cmap=None,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_states.set_ylabel("time step")
    ax_states.set_title(title or "XOR episode")

    inputs_img = inputs.T
    ax_inputs.set_facecolor(background_color)
    _plot_states(
        ax_inputs,
        inputs_img,
        cmap=None,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_inputs.set_xlabel("tick")
    ax_inputs.set_yticks(range(inputs_img.shape[0]))
    ax_inputs.set_yticklabels(["A", "B", "distractor", "cue"])
    ax_inputs.set_title("input channels")

    ax_inputs.axvline(output_tick, color="#ff7f0e", linewidth=2, alpha=0.8)
    if y_pred is None or y_pred < 0:
        pred_text = "pred: n/a"
    else:
        pred_text = f"pred: {y_pred}"
    ax_inputs.text(
        output_tick + 0.5,
        inputs_img.shape[0] - 0.5,
        f"true: {y_true} | {pred_text}",
        color="#000000",
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "#ffffff", "alpha": 0.8, "edgecolor": "none"},
    )

    plt.tight_layout()
    if show:
        plt.show()
    return ax_states, ax_inputs


def plot_xor_series(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    show_aux_channels: bool = False,
    title: str = "",
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.2,
    show: bool = True,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot XOR input/output series as timelines.

    inputs: shape (L, 4) with channels [A, B, distractor, cue]
    outputs: shape (L,) with values {0,1} in output window, else -1
    """
    fig, (ax_inputs, ax_outputs) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 4),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )

    channel_labels = ["A", "B"]
    channel_idx = [0, 1]
    if show_aux_channels:
        channel_labels = ["A", "B", "distractor", "cue"]
        channel_idx = [0, 1, 2, 3]

    inputs_img = inputs[:, channel_idx].T
    ax_inputs.set_facecolor(background_color)
    _plot_states(
        ax_inputs,
        inputs_img,
        cmap=None,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_inputs.set_ylabel("channels")
    ax_inputs.set_yticks(range(inputs_img.shape[0]))
    ax_inputs.set_yticklabels(channel_labels)
    ax_inputs.set_title(title or "XOR inputs/outputs")

    outputs_row = outputs.reshape(1, -1)
    out_cmap = ListedColormap(["#cccccc", "#d62728", "#2ca02c"])
    out_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], out_cmap.N)
    ax_outputs.set_facecolor(background_color)
    ax_outputs.imshow(
        outputs_row,
        cmap=out_cmap,
        norm=out_norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax_outputs.set_yticks([0])
    ax_outputs.set_yticklabels(["XOR"])
    ax_outputs.set_xlabel("tick")

    plt.tight_layout()
    if show:
        plt.show()
    return ax_inputs, ax_outputs


def plot_xor_summary(
    inputs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "",
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.2,
    show: bool = True,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """
    Plot side-by-side heatmaps: inputs, predictions, correctness.

    inputs: shape (L, 4) with channels [A, B, distractor, cue]
    y_true/y_pred: shape (L,), -1 outside output window
    """
    output_mask = y_true != -1
    if not np.any(output_mask):
        raise ValueError("y_true does not contain any output window values")

    out_idx = np.where(output_mask)[0]
    output_len = out_idx.size
    input_len = output_len

    inputs_a = inputs[:input_len, 0].reshape(1, -1)
    pred = y_pred[output_mask].reshape(1, output_len)
    corr = (y_pred[output_mask] == y_true[output_mask]).astype(np.int8).reshape(
        1, output_len
    )

    fig, (ax_in, ax_pred, ax_corr) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 3),
        gridspec_kw={"width_ratios": [2, 1, 1]},
    )

    ax_in.set_facecolor(background_color)
    _plot_states(
        ax_in,
        inputs_a,
        cmap=None,
        background_color=background_color,
        active_color=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_in.set_title("input (A)")
    ax_in.set_yticks([0])
    ax_in.set_yticklabels(["A"])
    ax_in.set_xlabel("bit index")

    pred_cmap = ListedColormap(["#d62728", "#2ca02c"])
    ax_pred.imshow(pred, cmap=pred_cmap, interpolation="nearest", aspect="auto")
    ax_pred.set_title("predicted XOR")
    ax_pred.set_yticks([0])
    ax_pred.set_yticklabels(["pred"])
    ax_pred.set_xlabel("bit index")

    corr_cmap = ListedColormap(["#d62728", "#2ca02c"])
    ax_corr.imshow(corr, cmap=corr_cmap, interpolation="nearest", aspect="auto")
    ax_corr.set_title("correct?")
    ax_corr.set_yticks([0])
    ax_corr.set_yticklabels(["match"])
    ax_corr.set_xlabel("bit index")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if show:
        plt.show()
    return ax_in, ax_pred, ax_corr


def plot_xor_batch_summary(
    inputs_a: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    title: str = "",
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.4,
    show: bool = True,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """
    Plot side-by-side heatmaps for XOR batch runs.

    inputs_a: shape (n_trials, n_bits)
    y_pred/y_true: shape (n_trials, n_bits)
    """
    if inputs_a.shape != y_pred.shape or y_pred.shape != y_true.shape:
        raise ValueError("inputs_a, y_pred, and y_true must have matching shapes")

    correct = (y_pred == y_true).astype(np.int8)

    fig, (ax_in, ax_pred, ax_corr) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    ax_in.set_facecolor(background_color)
    _plot_binary_grid(
        ax_in,
        inputs_a,
        color0=background_color,
        color1=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_in.set_title("inputs (A)")
    ax_in.set_xlabel("bit index")
    ax_in.set_ylabel("trial")

    ax_pred.set_facecolor(background_color)
    _plot_binary_grid(
        ax_pred,
        y_pred,
        color0=background_color,
        color1=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_pred.set_title("predicted XOR")
    ax_pred.set_xlabel("bit index")
    ax_pred.set_yticks([])

    ax_corr.set_facecolor(background_color)
    _plot_binary_grid(
        ax_corr,
        correct,
        color0="#d62728",
        color1="#2ca02c",
        border_color=border_color,
        border_width=border_width,
    )
    ax_corr.set_title("correct?")
    ax_corr.set_xlabel("bit index")
    ax_corr.set_yticks([])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if show:
        plt.show()
    return ax_in, ax_pred, ax_corr


def plot_xor_classification_summary(
    inputs: np.ndarray,
    y_pred_single: np.ndarray,
    y_true_single: np.ndarray,
    *,
    title: str = "",
    background_color: str = DEFAULT_BG_COLOR,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    correct_color: str = "#2ca02c",  # green
    wrong_color: str = "#d62728",  # red
    border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = 0.4,
    show: bool = True,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """
    Plot side-by-side heatmaps for XOR batch runs, for single-bit output.

    inputs: shape (n_trials, n_bits)
    y_pred_single/y_true_single: shape (n_trials,)
    """
    n_trials, n_bits = inputs.shape

    if y_pred_single.shape != (n_trials,) or y_true_single.shape != (n_trials,):
        raise ValueError(
            "y_pred_single and y_true_single must have shape (n_trials,)"
        )

    # Create the correctness matrix (n_trials, 1)
    correct = (y_pred_single == y_true_single).astype(np.int8)

    fig, (ax_in, ax_pred, ax_corr) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [n_bits, 1, 1]},
    )

    # Plot inputs
    ax_in.set_facecolor(background_color)
    _plot_binary_grid(
        ax_in,
        inputs,
        color0=background_color,
        color1=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_in.set_title("Inputs (k-bits)")
    ax_in.set_xlabel("Bit Index")
    ax_in.set_ylabel("Trial")

    # Plot predicted XOR (single bit)
    ax_pred.set_facecolor(background_color)
    _plot_binary_grid(
        ax_pred,
        y_pred_single.reshape(-1, 1),
        color0=background_color,
        color1=active_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_pred.set_title("Predicted XOR")
    ax_pred.set_xlabel("Output")
    ax_pred.set_yticks([])

    # Plot correctness
    ax_corr.set_facecolor(background_color)
    _plot_binary_grid(
        ax_corr,
        correct.reshape(-1, 1),
        color0=wrong_color,
        color1=correct_color,
        border_color=border_color,
        border_width=border_width,
    )
    ax_corr.set_title("Correct?")
    ax_corr.set_xlabel("Result")
    ax_corr.set_yticks([])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if show:
        plt.show()
    return ax_in, ax_pred, ax_corr


def _states_to_rgb(
    states: np.ndarray,
    *,
    cmap: Optional[str],
    background_color: str,
    active_color: str,
    border_color: str,
    cell_size: int,
    border_px: int,
) -> np.ndarray:
    """Convert binary states to an RGB image array with optional borders."""
    if cmap:
        cmap_fn = cm.get_cmap(cmap)
        normed = np.clip(states.astype(float), 0.0, 1.0)
        rgba = cmap_fn(normed)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return _scale_with_border(
            rgb,
            cell_size=cell_size,
            border_px=border_px,
            border_color=border_color,
        )

    bg_rgb = _hex_to_rgb(background_color)
    active_rgb = _hex_to_rgb(active_color)
    border_rgb = _hex_to_rgb(border_color)
    return _binary_to_rgb_with_border(
        states,
        bg_rgb=bg_rgb,
        active_rgb=active_rgb,
        border_rgb=border_rgb,
        cell_size=cell_size,
        border_px=border_px,
    )


def _pad_states(states: np.ndarray, target_height: int) -> np.ndarray:
    """Pad states with zeros on the bottom to a fixed height."""
    if states.shape[0] == target_height:
        return states
    if states.shape[0] > target_height:
        return states[-target_height:]
    pad = np.zeros((target_height - states.shape[0], states.shape[1]), dtype=states.dtype)
    return np.vstack([states, pad])


def _plot_states(
    ax: plt.Axes,
    states: np.ndarray,
    *,
    cmap: Optional[str],
    background_color: str,
    active_color: str,
    border_color: str,
    border_width: float,
) -> None:
    if cmap:
        ax.imshow(states, aspect="auto", interpolation="nearest", cmap=cmap)
        return
    colors = ListedColormap([background_color, active_color])
    height, width = states.shape
    x_edges = np.arange(0, width + 1)
    y_edges = np.arange(0, height + 1)
    ax.pcolormesh(
        x_edges,
        y_edges,
        states,
        cmap=colors,
        edgecolors=border_color,
        linewidth=border_width,
        shading="flat",
        antialiased=True,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def _plot_binary_grid(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    color0: str,
    color1: str,
    border_color: str,
    border_width: float,
) -> None:
    colors = ListedColormap([color0, color1])
    height, width = data.shape
    x_edges = np.arange(0, width + 1)
    y_edges = np.arange(0, height + 1)
    ax.pcolormesh(
        x_edges,
        y_edges,
        data,
        cmap=colors,
        vmin=0,
        vmax=1,
        edgecolors=border_color,
        linewidth=border_width,
        shading="flat",
        antialiased=True,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def _binary_to_rgb_with_border(
    states: np.ndarray,
    *,
    bg_rgb: np.ndarray,
    active_rgb: np.ndarray,
    border_rgb: np.ndarray,
    cell_size: int,
    border_px: int,
) -> np.ndarray:
    height, width = states.shape
    img = np.empty((height * cell_size, width * cell_size, 3), dtype=np.uint8)
    for r in range(height):
        for c in range(width):
            top = r * cell_size
            left = c * cell_size
            block = img[top : top + cell_size, left : left + cell_size]
            block[:] = active_rgb if states[r, c] else bg_rgb
            if border_px > 0:
                block[:border_px, :] = border_rgb
                block[-border_px:, :] = border_rgb
                block[:, :border_px] = border_rgb
                block[:, -border_px:] = border_rgb
    return img


def _scale_with_border(
    rgb: np.ndarray,
    *,
    cell_size: int,
    border_px: int,
    border_color: str,
) -> np.ndarray:
    height, width, _ = rgb.shape
    out = np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)
    if border_px <= 0:
        return out
    border_rgb = _hex_to_rgb(border_color)
    for r in range(height):
        for c in range(width):
            top = r * cell_size
            left = c * cell_size
            block = out[top : top + cell_size, left : left + cell_size]
            block[:border_px, :] = border_rgb
            block[-border_px:, :] = border_rgb
            block[:, :border_px] = border_rgb
            block[:, -border_px:] = border_rgb
    return out


def _ensure_figures_dir(fig_dir: str | None = None) -> str:
    path = fig_dir or "figures"
    os.makedirs(path, exist_ok=True)
    return path


def _sum_labels_for_pairs(n_bits: int) -> np.ndarray:
    sums = []
    for x in range(2**n_bits):
        for y in range(2**n_bits):
            sums.append(x + y)
    return np.array(sums, dtype=int)


def _reservoir_params_from_defaults() -> dict:
    return {
        "rule_number": 110,
        "width": 256,
        "boundary": "periodic",
        "recurrence": 8,
        "itr": 8,
        "d_period": 20,
        "repeats": 1,
        "seed": 0,
        "feature_mode": "cue_tick",
        "output_window": 2,
    }


def figure_full_accuracy_vs_n(fig_dir: str, n_list: list[int]) -> list[dict]:
    results = []
    direct_acc = []
    reservoir_acc = []

    params = _reservoir_params_from_defaults()

    for n_bits in n_list:
        n_samples = 2 ** (2 * n_bits)
        est_episodes = n_samples * n_bits * 5
        if est_episodes > 20000:
            print(f"Skipping N={n_bits} due to runtime estimate (episodes={est_episodes}).")
            continue

        X_direct, Y_direct = enumerate_addition_dataset(n_bits)
        res_direct = evaluate_linear_task(X_direct, Y_direct)

        X_res, Y_res = build_reservoir_dataset(n_bits, cin=0, **params)
        res_reservoir = evaluate_linear_task(X_res, Y_res)

        results.append(
            {
                "n_bits": n_bits,
                "direct": res_direct,
                "reservoir": res_reservoir,
            }
        )
        direct_acc.append(res_direct["full_acc"])
        reservoir_acc.append(res_reservoir["full_acc"])

    n_vals = [r["n_bits"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.plot(n_vals, direct_acc, marker="o", label="Direct")
    plt.plot(n_vals, reservoir_acc, marker="o", label="Reservoir")
    plt.xlabel("N bits")
    plt.ylabel("Full-vector accuracy")
    plt.title("Full-vector accuracy vs N")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_full_accuracy_vs_N.png"), dpi=200)
    plt.close()

    return results


def figure_per_bit_accuracy_and_balance(fig_dir: str, n_bits: int):
    params = _reservoir_params_from_defaults()

    X_direct, Y_direct = enumerate_addition_dataset(n_bits)
    res_direct = evaluate_linear_task(X_direct, Y_direct)

    X_res, Y_res = build_reservoir_dataset(n_bits, cin=0, **params)
    res_reservoir = evaluate_linear_task(X_res, Y_res)

    labels = [f"s{i}" for i in range(n_bits)] + ["cout"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - width, res_direct["per_bit_acc"], width, label="Direct")
    ax1.bar(x, res_reservoir["per_bit_acc"], width, label="Reservoir")
    ax1.bar(x + width, res_reservoir["majority_baseline_per_bit"], width, label="Majority")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(f"Per-bit accuracy and balance (N={n_bits})")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        res_reservoir["p1"],
        color="black",
        marker="o",
        linestyle="--",
        label="p(y=1)",
    )
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("p(y=1)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"fig2_per_bit_accuracy_and_balance_N{n_bits}.png"),
        dpi=200,
    )
    plt.close()


def figure_ablation_itr(fig_dir: str, n_bits: int, itr_list: list[int]):
    params = _reservoir_params_from_defaults()
    acc_full = []
    acc_per_bit_mean = []

    for itr in itr_list:
        params["itr"] = int(itr)
        X_res, Y_res = build_reservoir_dataset(n_bits, cin=0, **params)
        res = evaluate_linear_task(X_res, Y_res)
        acc_full.append(res["full_acc"])
        acc_per_bit_mean.append(float(np.mean(res["per_bit_acc"])))

    plt.figure(figsize=(6, 4))
    plt.plot(itr_list, acc_full, marker="o", label="Full-vector")
    plt.plot(itr_list, acc_per_bit_mean, marker="o", linestyle="--", label="Per-bit avg")
    plt.xlabel("itr")
    plt.ylabel("Accuracy")
    plt.title(f"Capacity ablation (itr), N={n_bits}")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"fig3_ablation_itr_N{n_bits}.png"), dpi=200)
    plt.close()


def figure_pca_direct_vs_reservoir(fig_dir: str, n_bits: int):
    params = _reservoir_params_from_defaults()

    X_direct, _ = enumerate_addition_dataset(n_bits)
    X_res, _ = build_reservoir_dataset(n_bits, cin=0, **params)
    sums = _sum_labels_for_pairs(n_bits)

    X_direct_centered = X_direct - X_direct.mean(axis=0, keepdims=True)
    X_res_centered = X_res - X_res.mean(axis=0, keepdims=True)

    pca_direct = PCA(n_components=2)
    pca_res = PCA(n_components=2)

    Z_direct = pca_direct.fit_transform(X_direct_centered)
    Z_res = pca_res.fit_transform(X_res_centered)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin = sums.min()
    vmax = sums.max()

    sc0 = axes[0].scatter(
        Z_direct[:, 0], Z_direct[:, 1], c=sums, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0].set_title("Direct PCA")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(Z_res[:, 0], Z_res[:, 1], c=sums, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reservoir PCA")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.colorbar(sc0, ax=axes, label="x + y")
    fig.suptitle(f"PCA of representations (N={n_bits})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"fig4_pca_direct_vs_reservoir_N{n_bits}.png"), dpi=200
    )
    plt.close()


def main() -> None:
    fig_dir = _ensure_figures_dir()

    n_list = [1, 2, 3, 4, 5]
    results = figure_full_accuracy_vs_n(fig_dir, n_list)

    with open(os.path.join(fig_dir, "results_sweep_N.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    figure_per_bit_accuracy_and_balance(fig_dir, n_bits=3)
    figure_ablation_itr(fig_dir, n_bits=4, itr_list=[1, 2, 4, 6, 8, 12, 16])
    figure_pca_direct_vs_reservoir(fig_dir, n_bits=3)

    print(f"Saved figures to {fig_dir}/")
