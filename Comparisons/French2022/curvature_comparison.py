"""Compare modeled and measured front-arrival times for SiO2_gold17.6."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATERIAL = "SiO2_gold17.6"
Z_DETECTOR_MM = 1.5
HEATMAP_TIME_NS = 4.5
ARTICLE_DATA_PATH = (
    PROJECT_ROOT
    / "Data_new"
    / "French2022"
    / MATERIAL
    / "article"
    / "fronts"
    / "curvature.csv"
)

# Select the intended material before importing the model and its parameters.
os.environ["PHYSICS_MATERIAL"] = MATERIAL

import parameters
from model_main import analytic_wave_front_dispatch
from radiation_flux import compute_and_plot_T4_heatmap


plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    }
)


def _sorted_snapshots(bessel_data):
    snapshots = []
    for key, snapshot in bessel_data.items():
        if isinstance(key, bool):
            continue
        try:
            time_ns = float(key)
        except (TypeError, ValueError):
            continue
        snapshots.append((time_ns, snapshot))

    snapshots.sort(key=lambda item: item[0])
    if not snapshots:
        raise ValueError("The model returned no numeric 2D snapshots.")
    return snapshots


def calculate_arrival_times(bessel_data, z_detector_cm, power_law=None):
    """Interpolate the time at which the radial front reaches the detector."""
    if power_law is None:
        power_law = (4 + parameters.alpha - parameters.beta) / 4

    snapshots = _sorted_snapshots(bessel_data)
    times_ns = np.asarray([time_ns for time_ns, _ in snapshots], dtype=float)
    r_cm = np.asarray(snapshots[0][1]["r_grid"], dtype=float)
    arrival_times_ns = np.full(r_cm.shape, np.nan, dtype=float)
    attenuation_factor = 1 - 0.065**power_law

    for radial_index in range(r_cm.size):
        front_positions_cm = np.asarray(
            [snapshot["z_F_radial"][radial_index] for _, snapshot in snapshots],
            dtype=float,
        )
        front_positions_cm = front_positions_cm * attenuation_factor
        front_positions_cm = np.maximum.accumulate(front_positions_cm)

        if z_detector_cm <= front_positions_cm[-1]:
            arrival_times_ns[radial_index] = np.interp(
                z_detector_cm,
                front_positions_cm,
                times_ns,
            )

    return r_cm, arrival_times_ns


def run_comparison():
    if not ARTICLE_DATA_PATH.exists():
        raise FileNotFoundError(f"Article curvature data not found: {ARTICLE_DATA_PATH}")

    article_data = pd.read_csv(ARTICLE_DATA_PATH)
    required_columns = {"x", "y"}
    if not required_columns.issubset(article_data.columns):
        raise ValueError(f"Article CSV must contain columns: {sorted(required_columns)}")

    times_ns = np.linspace(0.01, 8.0, 1600)
    result = analytic_wave_front_dispatch(
        times_ns,
        use_seconds=True,
        mode="marshak_ablation",
        vary_rho=True,
        lam_eff=True,
        power=0.6,
        wall_material="Gold",
    )
    bessel_data = result[5]

    power_law = (4 + parameters.alpha - parameters.beta) / 4
    r_cm, arrival_times_ns = calculate_arrival_times(
        bessel_data,
        Z_DETECTOR_MM / 10.0,
        power_law=power_law,
    )
    valid = np.isfinite(arrival_times_ns)
    if not np.any(valid):
        raise RuntimeError(
            f"The modeled front did not reach z = {Z_DETECTOR_MM:g} mm by {times_ns[-1]:g} ns."
        )

    r_model_mm = np.concatenate((-r_cm[valid][::-1], r_cm[valid][1:])) * 10.0
    t_model_ns = np.concatenate(
        (arrival_times_ns[valid][::-1], arrival_times_ns[valid][1:])
    )

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    axis.plot(
        r_model_mm,
        t_model_ns,
        color="firebrick",
        linewidth=2.5,
        label="Model",
    )
    axis.plot(
        article_data["x"],
        article_data["y"] - 0.15,
        linestyle=":",
        linewidth=2.0,
        color="black",
        alpha=0.9,
        label="Experiment",
        zorder=3,
    )
    axis.set_xlabel(r"Radial location $r$ [mm]")
    axis.set_ylabel(r"Front-arrival time $t_{\mathrm{arr}}$ [ns]")
    axis.set_xlim(-parameters.R_cm * 10.0, parameters.R_cm * 10.0)
    axis.set_ylim(3.0, 6.5)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=True)
    figure.tight_layout()

    figure_path = Path(__file__).resolve().parent / "curvature_comparison_1.5mm.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    # Use the project's standard heatmap implementation from radiation_flux.py.
    compute_and_plot_T4_heatmap(
        times_ns,
        mode="marshak_ablation",
        wall_material="Gold",
        use_seconds=True,
        vary_rho=True,
        lam_eff=True,
        power=1,
        show_plot=False,
        save_csv=True,
        time_snapshot_ns=HEATMAP_TIME_NS,
    )

    model_csv_path = Path(__file__).resolve().parent / "model_arrival_times_1.5mm.csv"
    pd.DataFrame(
        {
            "r_mm": r_model_mm,
            "arrival_time_ns": t_model_ns,
        }
    ).to_csv(model_csv_path, index=False)

    print(f"Saved comparison plot: {figure_path}")
    print(f"Saved heatmap using radiation_flux.py at t={HEATMAP_TIME_NS:.1f} ns")
    print(f"Saved model arrival times: {model_csv_path}")
    return figure_path, model_csv_path


if __name__ == "__main__":
    run_comparison()