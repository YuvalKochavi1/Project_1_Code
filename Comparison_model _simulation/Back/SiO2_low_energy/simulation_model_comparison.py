"""SiO2 low-energy (Back) comparison plots.

This script compares model curves for two drive profiles and overlays the gold 2D simulation:
- heyney
- flattop

It generates three figures (like the SiO2 comparison flow):
- front_position.png
- front_surface.png
- energy.png

Gold 2D simulation overlays are loaded from the low-energy data tree.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable when run from this folder
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import parameters as _parameters

_parameters.Material = "SiO2_low_energy"
_parameters.Experiment = "Back"

import albedo_model as _albedo_model
import model_main as _model_main
from csv_helpers import export_analytic_positions_csv
from model_main import BASE_DIR
from plot_helpers import compute_standard_analytic_front_series

MODES = ["heyney", "flattop"]
FRONT_TIMES_NS = [2.0, 6.0, 10.0]

MODE_STYLE = {
    "heyney": {"label": "Henyey", "color": "firebrick", "linestyle": "--", "lw": 2.4, "alt": "brown"},
    "flattop": {"label": "flattop", "color": "red", "linestyle": "-", "lw": 2.6, "alt": "crimson"},
}


def _set_solver_mode(mode_name):
    # Current model stack still has legacy bool checks in some paths.
    # Keep booleans here so heyney/flattop differ physically in all old call sites.
    if mode_name == "flattop":
        flag = True
    elif mode_name == "heyney":
        flag = False
    else:
        raise ValueError(f"Unsupported mode: {mode_name}")

    _parameters.Flattop_condition = flag
    _model_main.Flattop_condition = flag
    _albedo_model.Flattop_condition = flag


def _profile_suffix(mode_name):
    if mode_name == "flattop":
        return "_flattop"
    return ""


def _numeric_snapshot_keys(snapshot_dict):
    return sorted(
        [key for key in snapshot_dict.keys() if isinstance(key, (int, float, np.floating, np.integer))],
        key=float,
    )


def _closest_snapshot(snapshot_dict, target_time_ns):
    keys = _numeric_snapshot_keys(snapshot_dict)
    if not keys:
        raise ValueError("No numeric time keys found in bessel_data.")
    key_arr = np.asarray(keys, dtype=float)
    closest_key = float(key_arr[np.argmin(np.abs(key_arr - target_time_ns))])
    return closest_key, snapshot_dict[closest_key]


def _article_front_curves():
    base = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2_low_energy" / "article" / "fronts"

    def load_xy(name):
        df = pd.read_csv(base / name)
        return df["x"].to_numpy(dtype=float), 0.1 * df["y"].to_numpy(dtype=float)

    t_hr, x_hr = load_xy("HR.csv")
    t_1d, x_1d = load_xy("1D_model.csv")
    t_2d, x_2d = load_xy("2D_model.csv")
    t_exp, x_exp = load_xy("exp_results.csv")
    return (t_hr, x_hr), (t_1d, x_1d), (t_2d, x_2d), (t_exp, x_exp)


def _simulation_front_position():
    csv_path = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2_low_energy" / "2D_simulation" / "front_vs_time" / "front_position_vs_time_Gold_r0.csv"
    df = pd.read_csv(csv_path)
    required = {"time_ns", "front_position_cm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Gold front-position CSV must contain columns: {sorted(required)}")
    return {
        "time_ns": df["time_ns"].to_numpy(dtype=float),
        "front_position_cm": df["front_position_cm"].to_numpy(dtype=float),
    }


def _simulation_front_surface():
    csv_path = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2_low_energy" / "2D_simulation" / "front_surface" / "front_surface_profiles_gold.csv"
    df = pd.read_csv(csv_path)
    if "r_cm" not in df.columns:
        raise ValueError("Gold front-surface CSV must contain an 'r_cm' column")

    profiles = {}
    for col in df.columns:
        if col == "r_cm":
            continue
        if not col.startswith("zF_cm_t") or not col.endswith("ns"):
            continue
        t_ns = float(col[len("zF_cm_t") : -2])
        profiles[t_ns] = df[col].to_numpy(dtype=float)

    if not profiles:
        raise ValueError("No gold front-surface profiles found in the simulation CSV")

    return {"r_cm": df["r_cm"].to_numpy(dtype=float), "profiles": profiles}


def _simulation_energy():
    csv_path = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2_low_energy" / "2D_simulation" / "energy_comparison" / "simulated_energy_vs_time_gold.csv"
    df = pd.read_csv(csv_path)
    required = {"time_ns", "foam_energy_hJ", "gold_energy_hJ"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Gold energy CSV must contain columns: {sorted(required)}")
    return {
        "time_ns": df["time_ns"].to_numpy(dtype=float),
        "foam_energy_hJ": df["foam_energy_hJ"].to_numpy(dtype=float),
        "gold_energy_hJ": df["gold_energy_hJ"].to_numpy(dtype=float),
    }


def _compute_mode_series(times_ns, mode_name):
    _set_solver_mode(mode_name)
    front_series = compute_standard_analytic_front_series(times_ns, wall_material="Gold", lam_eff_power=1)

    model_dir = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2_low_energy" / "1.5 model"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_analytic_positions_csv(
        times_ns,
        {
            "front_position": {
                "Marshak": front_series["analytic_positions_marshak"],
                "Ablation with varying rho": front_series["analytic_positions_2D"],
                "2D effects + lam_eff": front_series["analytic_positions_2D_lam_eff"],
                "gold loss": front_series["analytic_positions_gold_loss"],
                "No Marshak": front_series["analytic_positions_no_marshak"],
            }
        },
        output_csv_path=model_dir / f"analytic_positions{_profile_suffix(mode_name)}.csv",
    )

    return {
        "mode": mode_name,
        "time_ns": np.asarray(times_ns, dtype=float),
        "x_gold_loss": np.asarray(front_series["analytic_positions_gold_loss"], dtype=float),
        "x_marshak": np.asarray(front_series["analytic_positions_marshak"], dtype=float),
        "E_marshak": np.asarray(front_series["E_marshak"], dtype=float),
        "E_gold_loss": np.asarray(front_series["E_gold_loss"], dtype=float),
        "E_wall_gold_loss": np.asarray(front_series["E_W_gold_loss"], dtype=float),
        "bessel_gold_loss": front_series["bessel_data_gold_loss"],
    }


def plot_front_position(output_dir, mode_data):
    (t_hr, x_hr), (t_1d, x_1d), (t_2d, x_2d), (t_exp, x_exp) = _article_front_curves()
    sim = _simulation_front_position()

    plt.figure(figsize=(8.2, 6.1))
    plt.plot(
        sim["time_ns"],
        sim["front_position_cm"],
        color="navy",
        linestyle="--",
        linewidth=2.5,
        label="Simulation (Gold)",
    )
    for entry in mode_data:
        style = MODE_STYLE[entry["mode"]]
        plt.plot(
            entry["time_ns"],
            entry["x_gold_loss"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["lw"],
            label=f"Model ({style['label']})",
        )

    # Article reference curves
    plt.plot(t_hr, x_hr, color="black", linestyle=":", linewidth=2.0, label="Article HR")
    plt.plot(t_1d, x_1d, color="black", linestyle="-.", linewidth=2.0, label="Article 1D")
    plt.plot(t_2d, x_2d, color="gray", linestyle="--", linewidth=2.0, label="Article 2D")
    plt.scatter(t_exp, x_exp, color="black", s=28, marker="o", label="Experiment")

    plt.xlabel(r"$t$ [ns]")
    plt.ylabel(r"$x_F$ [cm]")
    plt.ylim(0.0, 0.2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_path = output_dir / "front_position.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def plot_front_surface(output_dir, mode_data):
    sim = _simulation_front_surface()
    n_plots = len(FRONT_TIMES_NS)
    fig_height = 6.2
    axes_height = fig_height - 1.5
    axes_width = axes_height * (_parameters.R_cm / 0.2)
    wspace = 0.18
    fig_width = 0.9 + 0.3 + (n_plots + (n_plots - 1) * wspace) * axes_width

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={"wspace": wspace},
    )
    if n_plots == 1:
        axes = [axes]

    for idx, t_ns in enumerate(FRONT_TIMES_NS):
        ax = axes[idx]

        sim_profiles = sim["profiles"]
        if t_ns in sim_profiles:
            ax.plot(
                sim["r_cm"],
                sim_profiles[t_ns],
                color="navy",
                linestyle="--",
                linewidth=2.5,
                label=f"Simulation Gold t={t_ns:.1f} ns",
            )

        for entry in mode_data:
            style = MODE_STYLE[entry["mode"]]
            bessel = entry["bessel_gold_loss"]
            if not bessel:
                continue
            _, snap = _closest_snapshot(bessel, t_ns)
            r = np.asarray(snap.get("r_grid", []), dtype=float)
            zf = np.asarray(snap.get("z_F_radial", []), dtype=float)
            if r.size == 0 or zf.size == 0:
                continue

            ax.plot(
                r,
                zf,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["lw"],
                label=f"Model {style['label']} t={t_ns:.1f} ns",
            )

        ax.set_xlim([0, _parameters.R_cm])
        ax.set_ylim([0, 0.2])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$r$ [cm]")
        ax.legend(loc="best", fontsize=8)
        if idx == 0:
            ax.set_ylabel(r"$z_F$ [cm]")

    out_path = output_dir / "front_surface.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_energy(output_dir, mode_data):
    sim = _simulation_energy()
    plt.figure(figsize=(8.2, 6.1))

    plt.plot(
        sim["time_ns"],
        sim["foam_energy_hJ"],
        color="navy",
        linestyle="--",
        linewidth=2.5,
        label="Simulation Foam Energy",
    )
    plt.plot(
        sim["time_ns"],
        sim["gold_energy_hJ"],
        color="teal",
        linestyle=":",
        linewidth=2.5,
        label="Simulation Gold Energy",
    )

    # Marshak baseline from heyney run
    heyney_data = next((x for x in mode_data if x["mode"] == "heyney"), None)
    if heyney_data is not None:
        plt.plot(
            heyney_data["time_ns"],
            heyney_data["E_marshak"],
            color="orange",
            linestyle="-.",
            linewidth=2.8,
            label="Model E (Marshak)",
        )

    for entry in mode_data:
        style = MODE_STYLE[entry["mode"]]
        plt.plot(
            entry["time_ns"],
            entry["E_gold_loss"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["lw"],
            label=f"Model E Gold loss ({style['label']})",
        )
        plt.plot(
            entry["time_ns"],
            entry["E_wall_gold_loss"],
            color=style["alt"],
            linestyle=style["linestyle"],
            linewidth=style["lw"],
            label=f"Model E Gold wall loss ({style['label']})",
        )

    plt.xlabel(r"$t$ [ns]")
    plt.ylabel(r"$E$ [hJ]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_path = output_dir / "energy.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    times_ns = np.linspace(0.01, 15.0, 1000)
    mode_data = [_compute_mode_series(times_ns, mode_name) for mode_name in MODES]

    front_out = plot_front_position(out_dir, mode_data)
    surface_out = plot_front_surface(out_dir, mode_data)
    energy_out = plot_energy(out_dir, mode_data)

    print(f"Saved: {front_out}")
    print(f"Saved: {surface_out}")
    print(f"Saved: {energy_out}")


if __name__ == "__main__":
    main()
