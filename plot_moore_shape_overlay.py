"""Overlay model and Moore simulation shapes with y(r) / y(0) normalization.

Negative x coordinates are reflected to positive radius. If r=0 is missing,
the nearest-to-axis y value is used as y(0), with a constant extension to zero.

Run: python plot_moore_shape_overlay.py
Use --no-show to save the figure without opening a plot window.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from csv_helpers import read_xy_csv, save_figure_to_path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_new" / "Moore" / "SiO2_Moore" / "2D_shape"
OUTPUT_PATH = (
    BASE_DIR / "Figures_new" / "Moore" / "SiO2_Moore" / "2D_shape"
    / "model_vs_moore_normalized.png"
)


def load_normalized_shape(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read x/y data, reflect negative x, sort, and normalize at the axis."""
    x, y = read_xy_csv(path)
    radius = np.abs(np.asarray(x, dtype=float))
    values = np.asarray(y, dtype=float)
    if radius.size == 0:
        raise ValueError(f"{path}: no shape samples.")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: x and y must contain only finite numbers.")

    order = np.argsort(radius, kind="stable")
    radius, values = radius[order], values[order]
    center_value = values[0]
    if center_value == 0:
        raise ValueError(f"{path}: cannot normalize a zero center value.")
    values = values / center_value
    if radius[0] > 0:
        print(
            f"{path.name}: no r=0 sample; using y={center_value:.8g} "
            f"at |r|={radius[0]:.8g} as the center value."
        )
        radius = np.insert(radius, 0, 0.0)
        values = np.insert(values, 0, 1.0)
    return radius, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window.")
    args = parser.parse_args()

    model_r, model_y = load_normalized_shape(DATA_DIR / "model.csv")
    moore_r, moore_y = load_normalized_shape(DATA_DIR / "moore_simulation.csv")

    with plt.rc_context({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 16,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "axes.linewidth": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(model_r, model_y, color="#0072B2", linewidth=2.2, label="Model")
        ax.plot(moore_r, moore_y, color="#D55E00", linewidth=2.2, linestyle="--",
                label="Moore simulation")
        ax.set(
            xlabel=r"$r$ [cm]",
            ylabel=r"$x_F(r)/x_F(0)$",
            xlim=(0, None),
        )
        ax.grid(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure_to_path(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {args.output.resolve()}")
        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
