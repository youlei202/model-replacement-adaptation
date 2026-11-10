from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7A7A7A",
}

ADAPTER_COLORS = {
    "mean_all": OKABE_ITO["grey"],
    "fixed_peer": OKABE_ITO["grey"],
    "fixed_pair_mean": "#A0A0A0",
    "val_select_pair_mean": "#4D4D4D",
    "convex": OKABE_ITO["blue"],
    "linear": OKABE_ITO["green"],
    "rff": OKABE_ITO["orange"],
    "krr": OKABE_ITO["vermillion"],
    "mlp": OKABE_ITO["purple"],
}


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(False)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.65, alpha=0.65)
    ax.set_axisbelow(True)


def label_panels(axes: Iterable, start: str = "A") -> None:
    for offset, ax in enumerate(np.asarray(list(axes), dtype=object).flat):
        label = chr(ord(start) + offset)
        ax.text(
            -0.13, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left",
        )


def bootstrap_interval(
    values: Sequence[float], statistic=np.median, draws: int = 2000, seed: int = 20260722,
) -> Tuple[float, float, float]:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        return np.nan, np.nan, np.nan
    center = float(statistic(sample))
    if len(sample) == 1:
        return center, center, center
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(sample), size=(draws, len(sample)))
    estimates = np.asarray([statistic(sample[index]) for index in indices], dtype=np.float64)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return center, float(lower), float(upper)


def save_publication_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", facecolor="white")
