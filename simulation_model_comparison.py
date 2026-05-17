"""Compare front positions and shapes between 2D simulation and analytical model.

This script loads front data from:
  1. 2D simulation: front surface profiles and positions
  2. Analytical model: front shapes from shape_2D_analytical_model.py
  
And compares them using plots and statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

from parameters import Experiment, Material, R_cm, L
from model_main import BASE_DIR


# Set up figures output directory
FIGURES_OUTPUT_DIR = Path(BASE_DIR) / "Figures_new" / Experiment / Material / "comparison_model_simulation"
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulation_front_surface_load():
  """Load 2D simulation front-surface profiles from CSV.

  Expected CSV header format:
    r_cm,zF_cm_t1.0ns,zF_cm_t2.0ns,zF_cm_t2.5ns

  Returns:
    dict with keys:
      "r_cm": numpy array of radial coordinates
      "profiles": dict mapping time in ns (float) -> z_F profile array
      "columns": dict mapping original column name -> z_F profile array
  """
  csv_path = (
    Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_surface" / "front_surface_profiles.csv"
  )

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

    match = re.match(r"zF_cm_t([\d.]+)ns$", col)
    if not match:
      continue

    time_ns = float(match.group(1))
    z_profile = df[col].to_numpy()
    profiles[time_ns] = z_profile
    columns[col] = z_profile

  if not profiles:
    raise ValueError(
      "No front-surface profile columns found. Expected names like 'zF_cm_t1.0ns'."
    )

  return {
    "r_cm": r_cm,
    "profiles": profiles,
    "columns": columns,
  }


def model_front_shape_load():
  """Load analytical-model front shapes from 3 CSV files.

  Expected files:
    front_shape_t0.50ns.csv
    front_shape_t1.00ns.csv
    front_shape_t1.30ns.csv

  Returns:
    dict with keys:
      "times_ns": sorted list of times in ns
      "profiles": dict mapping time in ns (float) -> dict with:
        "r_cm": radial coordinates
        "z_F_radial_cm": model front profile
      "files": dict mapping original filename -> loaded profile dict
  """
  data_dir = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape"
  if Material == "C8H8":
    expected_files = [
      "front_shape_t0.50ns.csv",
      "front_shape_t1.00ns.csv",
      "front_shape_t1.30ns.csv",
    ]
  elif Material == "SiO2":
    expected_files = [
      "front_shape_t1.00ns.csv",
      "front_shape_t2.00ns.csv",
      "front_shape_t2.50ns.csv",
    ]
  elif Material == "Ta2O5":
    expected_files = [
      "front_shape_t1.00ns.csv",
      "front_shape_t2.00ns.csv",
      "front_shape_t2.50ns.csv",
    ]

  profiles = {}
  files = {}

  for filename in expected_files:
    csv_path = data_dir / filename
    if not csv_path.exists():
      raise FileNotFoundError(f"Model front-shape CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"r_cm", "z_F_radial_cm"}
    if not required_cols.issubset(df.columns):
      raise ValueError(
        f"{filename} must contain columns: r_cm, z_F_radial_cm"
      )

    match = re.search(r"t([\d.]+)ns", filename)
    if not match:
      raise ValueError(f"Could not parse time from filename: {filename}")

    time_ns = float(match.group(1))
    entry = {
      "r_cm": df["r_cm"].to_numpy(),
      "z_F_radial_cm": df["z_F_radial_cm"].to_numpy(),
    }

    profiles[time_ns] = entry
    files[filename] = entry

  return {
    "times_ns": sorted(profiles.keys()),
    "profiles": profiles,
    "files": files,
  }


def plot_simulation_vs_model_each_time(sim_data, model_data):
  """Create one figure comparing simulation vs model at each model time."""
  sim_r = sim_data["r_cm"]
  sim_profiles = sim_data["profiles"]
  model_times = model_data["times_ns"]
  model_profiles = model_data["profiles"]

  if not model_times:
    raise ValueError("No model times found for plotting")

  available_sim_times = np.array(sorted(sim_profiles.keys()), dtype=float)
  
  # Filter model times to only those with a close simulation match (within 0.5 ns)
  matched_model_times = []
  for t_model in model_times:
    min_dist = np.min(np.abs(available_sim_times - t_model))
    if min_dist <= 0.5:
      matched_model_times.append(t_model)
  
  if not matched_model_times:
    raise ValueError("No model times with matching simulation data (within 0.2 ns tolerance)")
  
  model_times = matched_model_times
  n_times = len(model_times)
  fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 4.8), sharey=True)
  if n_times == 1:
    axes = [axes]

  for idx, t_model in enumerate(model_times):
    ax = axes[idx]
    model_entry = model_profiles[t_model]
    r_model = model_entry["r_cm"]
    z_model = model_entry["z_F_radial_cm"]

    # Match to the earliest simulation time >= model time (forward-matching).
    future_times = available_sim_times[available_sim_times >= t_model]
    if len(future_times) > 0:
      t_sim = float(future_times[0])  # First simulation time at or after model time
    else:
      # Fallback: use closest if no future time available
      t_sim = float(available_sim_times[np.argmin(np.abs(available_sim_times - t_model))])
    z_sim = sim_profiles[t_sim]

    ax.plot(r_model, z_model, color="tab:red", linewidth=2.5, label=f"Model t={t_model:.2f} ns")
    ax.plot(sim_r, z_sim, color="tab:blue", linestyle="--", linewidth=2.2, label=f"Simulation t={t_sim:.2f} ns")

    ax.set_xlim([0, R_cm])
    ax.set_ylim([0, L])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("r (cm)")
    ax.set_title(f"Model {t_model:.2f} ns")
    if idx == 0:
      ax.set_ylabel("z_F (cm)")
      ax.legend(loc="best", fontsize=9)

  fig.suptitle(
    f"Simulation vs Model Front Comparison at Each Time\n({Experiment} - {Material})",
    fontsize=13,
  )
  fig.tight_layout()

  out_path = FIGURES_OUTPUT_DIR / "simulation_vs_model_each_time.png"
  fig.savefig(out_path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return out_path


def simulation_front_position_load():
  """Load Ji-Yan 2D simulation front-position data from CSV.

  Expected CSV header format:
    time_ns,front_position_mm,front_position_cm

  Returns:
    dict with keys:
      "time_ns": numpy array of time values
      "front_position_mm": numpy array of front positions in mm
      "front_position_cm": numpy array of front positions in cm
  """
  csv_path = (
    Path(BASE_DIR)
    / "Data_new"
    / Experiment
    / Material
    / "2D_simulation"
    / "front_vs_time"
    / "front_position_vs_time_r0.csv"
  )

  if not csv_path.exists():
    raise FileNotFoundError(f"Front-position CSV not found: {csv_path}")

  df = pd.read_csv(csv_path)

  required_columns = {"time_ns", "front_position_mm", "front_position_cm"}
  missing_columns = required_columns.difference(df.columns)
  if missing_columns:
    raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")

  return {
    "time_ns": df["time_ns"].to_numpy(dtype=float),
    "front_position_mm": df["front_position_mm"].to_numpy(dtype=float),
    "front_position_cm": df["front_position_cm"].to_numpy(dtype=float),
  }


def model_front_position_load():
  """Load Ji-Yan analytical front-position comparison data from CSV.

  Expected CSV header format:
    time_ns,front_position (Marshak),front_position (Vacuum Loss)

  Returns:
    dict with keys:
      "time_ns": numpy array of time values
      "front_position_marshak_cm": numpy array of Marshak front positions in cm
      "front_position_vacuum_cm": numpy array of vacuum-loss front positions in cm
  """
  if Material == "C8H8":
    csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "1.5 model" / "analytic_positions_ji_yan.csv"

    if not csv_path.exists():
      raise FileNotFoundError(f"Model front-position CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"time_ns", "front_position (Marshak)", "front_position (Vacuum Loss)"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
      raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")

    return {
      "time_ns": df["time_ns"].to_numpy(dtype=float),
      "front_position_marshak_cm": df["front_position (Marshak)"].to_numpy(dtype=float),
      "front_position_vacuum_cm": df["front_position (Vacuum Loss)"].to_numpy(dtype=float),
    }
  elif Material == "SiO2":
    csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "1.5 model" / "analytic_positions.csv"
    if not csv_path.exists():
      raise FileNotFoundError(f"Model front-position CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_columns = {"time_ns", "front_position (Marshak)", "front_position (Gold Loss)"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
      raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")
    return {
      "time_ns": df["time_ns"].to_numpy(dtype=float),
      "front_position_marshak_cm": df["front_position (Marshak)"].to_numpy(dtype=float),
      "front_position_gold_cm": df["front_position (Gold Loss)"].to_numpy(dtype=float),
    }
  elif Material == "Ta2O5":
    csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "1.5 model" / "analytic_positions.csv"
    if not csv_path.exists():
      raise FileNotFoundError(f"Model front-position CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_columns = {"time_ns", "front_position (Marshak)", "front_position (2D effects)", "front_position (2D effects + lam_eff)", "front_position (Be Loss)"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
      raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")
    return {
      "time_ns": df["time_ns"].to_numpy(dtype=float),
      "front_position_marshak_cm": df["front_position (Marshak)"].to_numpy(dtype=float),
      "front_position_2D_cm": df["front_position (2D effects)"].to_numpy(dtype=float),
      "front_position_2D_lam_eff_cm": df["front_position (2D effects + lam_eff)"].to_numpy(dtype=float),
      "front_position_be_loss_cm": df["front_position (Be Loss)"].to_numpy(dtype=float),
    }
    

def plot_simulation_vs_model_front_position(sim_data, model_data):
  """Create a front-position-vs-time comparison for the Ji-Yan case."""
  plt.figure(figsize=(8, 6))

  plt.plot(
    sim_data["time_ns"],
    sim_data["front_position_cm"],
    color="tab:blue",
    linestyle="--",
    linewidth=2.2,
    label="2D simulation front (r=0)",
  )
  plt.plot(
    model_data["time_ns"],
    model_data["front_position_marshak_cm"],
    color="tab:red",
    linestyle="-",
    linewidth=2.4,
    label="Analytic model (Marshak)",
  )
  if Material == "SiO2":
    plt.plot(
      model_data["time_ns"],
      model_data["front_position_gold_cm"],
      color="tab:green",
      linestyle="-.",
      linewidth=2.2,
      label="Analytic model (Gold Loss)",
    )
  elif Material == "C8H8":
    plt.plot(
      model_data["time_ns"],
      model_data["front_position_vacuum_cm"],
      color="tab:green",
      linestyle="-.",
      linewidth=2.2,
      label="Analytic model (Vacuum Loss)",
    )
  elif Material == "Ta2O5":
    plt.plot(
      model_data["time_ns"],
      model_data["front_position_2D_cm"],
      color="tab:orange",
      linestyle="-.",
      linewidth=2.2,
      label="Analytic model (2D effects)",
    )
    plt.plot(
      model_data["time_ns"],
      model_data["front_position_2D_lam_eff_cm"],
      color="tab:purple",
      linestyle=":",
      linewidth=2.2,
      label="Analytic model (2D effects + lam_eff)",
    )
    plt.plot(
      model_data["time_ns"],
      model_data["front_position_be_loss_cm"],
      color="tab:green",
      linestyle="-.",
      linewidth=2.2,
      label="Analytic model (Be Loss)",
    )

  plt.xlabel("Time (ns)")
  plt.ylabel("Front position (cm)")
  plt.title(f"Front Position vs Time ({Experiment} - {Material})")
  plt.grid(True, alpha=0.3)
  plt.legend(loc="best")
  plt.tight_layout()

  out_path = FIGURES_OUTPUT_DIR / "simulation_vs_model_front_position.png"
  plt.savefig(out_path, dpi=180, bbox_inches="tight")
  plt.close()
  return out_path


def main():
  sim_data = simulation_front_surface_load()
  model_data = model_front_shape_load()
  out_path = plot_simulation_vs_model_each_time(sim_data, model_data)
  print(f"Saved comparison figure: {out_path}")

  front_position_sim_data = simulation_front_position_load()
  front_position_model_data = model_front_position_load()
  front_position_out_path = plot_simulation_vs_model_front_position(
    front_position_sim_data,
    front_position_model_data,
  )
  print(f"Saved front-position comparison figure: {front_position_out_path}")


if __name__ == "__main__":
  main()

