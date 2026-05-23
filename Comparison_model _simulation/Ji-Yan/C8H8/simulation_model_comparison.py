"""Ji-Yan (C8H8) — compare 2D simulation to analytic model.

This file is C8H8-only: all branches and references to other
materials/variants were removed.
"""

import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is importable when run from this folder
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import parameters as _parameters
_parameters.Material = "C8H8"
_parameters.Experiment = "Ji-Yan"
importlib.reload(_parameters)

from parameters import Experiment, Material, R_cm, L
from model_main import BASE_DIR

print(f"Experiment: {Experiment}, Material: {Material}")

# Output directory for figures of this comparison
FIGURES_OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulation_front_surface_load():
    """Load 2D simulation front-surface profiles.

    Expects CSV with columns like: r_cm, zF_cm_t1.00ns, zF_cm_t2.00ns
    Returns dict: {r_cm, profiles(dict time_ns->z_profile), columns(dict)}
    """
    csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_surface" / "front_surface_profiles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Front-surface CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "r_cm" not in df.columns:
        raise ValueError("CSV must contain 'r_cm' column")
    r_cm = df["r_cm"].to_numpy()
    profiles = {}
    columns = {}
    for col in df.columns:
        if col == "r_cm":
            continue
        m = re.match(r"zF_cm_t([\d.]+)ns$", col)
        if not m:
            continue
        t = float(m.group(1))
        profiles[t] = df[col].to_numpy()
        columns[col] = df[col].to_numpy()
    if not profiles:
        raise ValueError("No front-surface profile columns found. Expected names like 'zF_cm_t1.00ns'.")
    return {"r_cm": r_cm, "profiles": profiles, "columns": columns}


def model_front_shape_load():
    """Discover and load analytic front-shape CSVs for Ji-Yan.

    Finds any files matching `front_shape_t*.csv` in the folder and loads
    their `r_cm` and `z_F_radial_cm` columns, returning a time-sorted map.
    """
    data_dir = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape"
    if not data_dir.exists():
        raise FileNotFoundError(f"Model shape directory not found: {data_dir}")
    profiles = {}
    files = {}
    for p in sorted(data_dir.glob("front_shape_t*ns.csv")):
        df = pd.read_csv(p)
        required = {"r_cm", "z_F_radial_cm"}
        if not required.issubset(df.columns):
            continue
        m = re.search(r"t([\d.]+)ns", p.name)
        if not m:
            continue
        t = float(m.group(1))
        entry = {"r_cm": df["r_cm"].to_numpy(), "z_F_radial_cm": df["z_F_radial_cm"].to_numpy()}
        profiles[t] = entry
        files[p.name] = entry
    if not profiles:
        raise FileNotFoundError(f"No model front-shape CSVs found in {data_dir}")
    return {"times_ns": sorted(profiles.keys()), "profiles": profiles, "files": files}


def plot_simulation_vs_model_each_time(sim_data, model_data, output_dir):
    """Create one figure (columns per model time) comparing sim vs model."""
    sim_r = sim_data["r_cm"]
    sim_profiles = sim_data["profiles"]
    model_times = model_data["times_ns"]
    model_files = model_data.get("files", {})

    available_sim_times = np.array(sorted(sim_profiles.keys()), dtype=float)
    # keep only model times that are within 0.5 ns of a simulation snapshot
    model_times = [t for t in model_times if np.min(np.abs(available_sim_times - t)) <= 0.5]
    if not model_times:
        raise ValueError("No model times with matching simulation data (within 0.5 ns tolerance)")

    n_times = len(model_times)
    fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 4.8), sharey=True)
    if n_times == 1:
        axes = [axes]

    for idx, t_model in enumerate(model_times):
        ax = axes[idx]
        future = available_sim_times[available_sim_times >= t_model]
        t_sim = float(future[0]) if len(future) > 0 else float(available_sim_times[np.argmin(np.abs(available_sim_times - t_model))])
        z_sim = sim_profiles[t_sim]
        ax.plot(sim_r, z_sim, color="tab:blue", linestyle="--", linewidth=2.2, label=f"Simulation t={t_sim:.2f} ns")
        # find matching model file entry
        matched = None
        for name, entry in model_files.items():
            m = re.search(r"t([\d.]+)ns", name)
            if not m:
                continue
            if abs(float(m.group(1)) - t_model) < 1e-6:
                matched = entry
                break
        if matched is not None:
            ax.plot(matched["r_cm"], matched["z_F_radial_cm"], color="tab:red", linewidth=2.5, label="Analytic model")
        ax.set_xlim([0, R_cm])
        ax.set_ylim([0, L])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("r (cm)")
        ax.set_title(f"Model {t_model:.2f} ns")
        if idx == 0:
            ax.set_ylabel("z_F (cm)")
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Simulation vs Model Front Comparison\n({Experiment} - {Material})", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "simulation_vs_model_each_time.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def simulation_front_position_load():
    """Load Ji-Yan 2D simulation front-position data (r=0)."""
    csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_vs_time" / "front_position_vs_time_r0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Front-position CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"time_ns", "front_position_mm", "front_position_cm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    return {"time_ns": df["time_ns"].to_numpy(dtype=float), "front_position_mm": df["front_position_mm"].to_numpy(dtype=float), "front_position_cm": df["front_position_cm"].to_numpy(dtype=float)}


def model_front_position_load():
    """Load Ji-Yan analytic front-position CSV (Marshak + Vacuum Loss)."""
    csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "1.5 model" / "analytic_positions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Model front-position CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"time_ns", "front_position (Marshak)", "front_position (Vacuum Loss)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    return {"time_ns": df["time_ns"].to_numpy(dtype=float), "front_position_marshak_cm": df["front_position (Marshak)"].to_numpy(dtype=float), "front_position_vacuum_cm": df["front_position (Vacuum Loss)"].to_numpy(dtype=float)}


def plot_simulation_vs_model_front_position(sim_data, model_data, output_dir):
    """Plot front-position vs time: Marshak and Vacuum Loss comparisons."""
    sim_t = sim_data["time_ns"]
    sim_fp = sim_data["front_position_cm"]
    x = model_data["time_ns"]
    series = [
        ("Analytic model (Marshak)", model_data["front_position_marshak_cm"], {"color": "tab:red", "linestyle": "-", "linewidth": 2.4}),
        ("Analytic model (Vacuum Loss)", model_data["front_position_vacuum_cm"], {"color": "tab:green", "linestyle": "-.", "linewidth": 2.2}),
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(sim_t, sim_fp, color="tab:blue", linestyle="--", linewidth=2.2, label="2D simulation front (r=0)")
    for label, y_arr, opts in series:
        plt.plot(x, y_arr, label=label, **opts)
    plt.xlabel("Time (ns)")
    plt.ylabel("Front position (cm)")
    plt.title(f"Front Position vs Time ({Experiment} - {Material})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = output_dir / "simulation_vs_model_front_position_standard.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return [out_path]


def main():
    # Vacuum-only run: prefer vacuum-specific CSVs if present, otherwise fall back
    # to the generic loaders.
    out_dir = FIGURES_OUTPUT_DIR / "Vacuum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # front-surface
    sim_fs_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_surface" / "front_surface_profiles_vacuum.csv"
    if sim_fs_path.exists():
        df = pd.read_csv(sim_fs_path)
        # build sim_data structure expected by plotting function
        r_cm = df["r_cm"].to_numpy()
        profiles = {}
        for col in df.columns:
            if col == "r_cm":
                continue
            m = re.match(r"zF_cm_t([\d.]+)ns$", col)
            if not m:
                continue
            profiles[float(m.group(1))] = df[col].to_numpy()
        sim_data = {"r_cm": r_cm, "profiles": profiles}
    else:
        sim_data = simulation_front_surface_load()

    model_shape = model_front_shape_load()
    shape_out = plot_simulation_vs_model_each_time(sim_data, model_shape, out_dir)
    print(f"Saved comparison figure(s): {shape_out}")

    # front-position (r=0)
    sim_fp_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_vs_time" / "front_position_vs_time_vacuum_r0.csv"
    if sim_fp_path.exists():
        dfp = pd.read_csv(sim_fp_path)
        front_sim = {"time_ns": dfp["time_ns"].to_numpy(dtype=float), "front_position_cm": dfp["front_position_cm"].to_numpy(dtype=float)}
    else:
        front_sim = simulation_front_position_load()

    front_model = model_front_position_load()
    pos_out = plot_simulation_vs_model_front_position(front_sim, front_model, out_dir)
    print(f"Saved front-position comparison figure(s): {pos_out}")

    # energy (vacuum)
    sim_energy_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "energy_comparison" / "simulated_energy_vs_time_vacuum.csv"
    if sim_energy_path.exists():
        df_e = pd.read_csv(sim_energy_path)
        sim_energy = {"time_ns": df_e["time_ns"].to_numpy(), "foam_energy_hJ": df_e["foam_energy_hJ"].to_numpy(), "coating_energy_hJ": np.zeros_like(df_e["time_ns"].to_numpy())}
    else:
        # fallback to a generic energy file if present
        generic_energy = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "energy_comparison" / "simulated_energy_vs_time.csv"
        if generic_energy.exists():
            df_e = pd.read_csv(generic_energy)
            sim_energy = {"time_ns": df_e["time_ns"].to_numpy(), "foam_energy_hJ": df_e["foam_energy_hJ"].to_numpy(), "coating_energy_hJ": df_e.get("gold_energy_hJ", np.zeros_like(df_e["time_ns"].to_numpy()))}
        else:
            sim_energy = {"time_ns": np.array([]), "foam_energy_hJ": np.array([]), "coating_energy_hJ": np.array([])}

    # Load analytical model energy data if present
    model_energy_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape" / f"simulated_energy_vs_time_{Material}_vacuum_loss.csv"
    model_energy = None
    if model_energy_path.exists():
        df_me = pd.read_csv(model_energy_path)
        if "time_ns" in df_me.columns and "E_marshak" in df_me.columns and "E_vacuum_loss" in df_me.columns:
            model_energy = {
                "time_ns": df_me["time_ns"].to_numpy(),
                "E_marshak": df_me["E_marshak"].to_numpy(),
                "E_vacuum_loss": df_me["E_vacuum_loss"].to_numpy()
            }

    if sim_energy["time_ns"].size > 0:
        # energy plot comparing simulation and analytic model
        import matplotlib.pyplot as _plt

        _plt.figure(figsize=(8, 6))
        _plt.plot(sim_energy["time_ns"], sim_energy["foam_energy_hJ"], color="tab:blue", linestyle="--", linewidth=2.2, label="2D simulation foam energy")
        
        if model_energy is not None:
            _plt.plot(model_energy["time_ns"], model_energy["E_marshak"], color="tab:red", linestyle="-", linewidth=2.4, label="Analytic model (Marshak)")
            _plt.plot(model_energy["time_ns"], model_energy["E_vacuum_loss"], color="tab:green", linestyle="-.", linewidth=2.2, label="Analytic model (Vacuum Loss)")

        _plt.xlabel("Time (ns)")
        _plt.ylabel("Energy (hJ)")
        _plt.title(f"Energy (Vacuum) ({Experiment} - {Material})")
        _plt.grid(True, alpha=0.3)
        _plt.legend(loc="best")
        _plt.tight_layout()
        e_out = out_dir / "energy_vacuum.png"
        _plt.savefig(e_out, dpi=180, bbox_inches="tight")
        _plt.close()
        print(f"Saved energy figure: {e_out}")


if __name__ == "__main__":
    main()
