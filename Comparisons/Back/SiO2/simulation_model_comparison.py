"""SiO2 (Back) comparison plots for Be and Gold.

This script writes one set of figures to `Be/` and one set to `Gold/`.
"""

import re
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

_parameters.Material = "SiO2"
_parameters.Experiment = "Back"


from model_main import BASE_DIR
from parameters import Experiment, L, R_cm, Material

print(f"Experiment: {Experiment}, Material: {Material}")

FRONT_TIMES_NS = [1, 2, 2.5]
COMBINED_TIME_NS = 2.5
COATING_COLORS = {
    "Be": "tab:blue",
    "Gold": "tab:orange",
    "Gold_100": "tab:red",
    "Vacuum": "tab:green",
    "Copper": "tab:purple",
}

plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'axes.unicode_minus': False,
    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.linewidth': 2.0,
    'font.size': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
})

def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def pick_column(df, *wanted_names):
    normalized = {normalize_name(column): column for column in df.columns}
    for wanted in wanted_names:
        key = normalize_name(wanted)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of columns: {wanted_names}")


def resolve_profile_time_key(profiles, requested_time_ns, tol_ns=1e-6):
    """Resolve time key robustly across int/float/string and tiny precision drifts."""
    if requested_time_ns in profiles:
        return requested_time_ns

    requested = float(requested_time_ns)
    available = np.array([float(key) for key in profiles.keys()], dtype=float)
    idx = int(np.argmin(np.abs(available - requested)))
    closest = float(available[idx])
    if abs(closest - requested) > tol_ns:
        raise KeyError(
            f"Requested profile time {requested_time_ns} ns not found. "
            f"Available times: {sorted(available.tolist())}"
        )

    for key in profiles.keys():
        if abs(float(key) - closest) <= tol_ns:
            return key

    raise KeyError(
        f"Requested profile time {requested_time_ns} ns could not be resolved from available keys."
    )


def choose_nearest_time(available_times, requested_time_ns):
    if not available_times:
        raise ValueError("No available times found for combined coating plot.")
    requested = float(requested_time_ns)
    nearest = min(available_times, key=lambda t: abs(float(t) - requested))
    if abs(float(nearest) - requested) > 1e-6:
        print(
            f"[Combined] requested time {requested_time_ns:.1f} ns not available; "
            f"using nearest common time {float(nearest):.2f} ns."
        )
    return float(nearest)


def wall_configurations():
    return [
        {
            "name": "Be",
            "front_loss_column": "front_position (Be Loss)",
            "simulation_front_position": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "front_vs_time"
            / "front_position_vs_time_Be_r0.csv",
            "simulation_front_surface": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "front_surface"
            / "front_surface_profiles_be.csv",
            "simulation_energy": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "energy_comparison"
            / "simulated_energy_vs_time_be.csv",
            "output_dir": Path(__file__).resolve().parent / "Be",
        },
        {
            "name": "Gold",
            "front_loss_column": "front_position (gold loss)",
            "simulation_front_position": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "front_vs_time"
            / "front_position_vs_time_Gold_r0.csv",
            "simulation_front_surface": Path(BASE_DIR)
            / "Data_new"
            / "Back" 
            / "SiO2"
            / "2D_simulation"
            / "front_surface"
            / "front_surface_profiles_gold.csv",
            "simulation_energy": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "energy_comparison"
            / "simulated_energy_vs_time_gold.csv",
            "output_dir": Path(__file__).resolve().parent / "Gold",
        },
        {
            "name": "Gold_100",
            "front_loss_column": "front_position (gold loss)",
            "simulation_front_position": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation_100"
            / "front_vs_time"
            / "front_position_vs_time_Gold_r0.csv",
            "simulation_front_surface": Path(BASE_DIR)
            / "Data_new"
            / "Back" 
            / "SiO2"
            / "2D_simulation_100"
            / "front_surface"
            / "front_surface_profiles_gold.csv",
            "simulation_energy": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation_100"
            / "energy_comparison"
            / "simulated_energy_vs_time_gold.csv",
            "output_dir": Path(__file__).resolve().parent / "Gold_100",
        },
        {
            "name": "Vacuum",
            "front_loss_column": "front_position (Vacuum loss)",
            "simulation_front_position": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "front_vs_time"
            / "front_position_vs_time_Vacuum_r0.csv",
            "simulation_front_surface": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "front_surface"
            / "front_surface_profiles_vacuum.csv",
            "simulation_energy": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "energy_comparison"
            / "simulated_energy_vs_time_vacuum.csv",
            "output_dir": Path(__file__).resolve().parent / "Vacuum",
        },
        {
            "name": "Copper",
            "front_loss_column": "front_position (Copper loss)",
            "simulation_front_position": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "front_vs_time"
            / "front_position_vs_time_Copper_r0.csv",
            "simulation_front_surface": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "front_surface"
            / "front_surface_profiles_copper.csv",
            "simulation_energy": Path(BASE_DIR)
            / "Data_new"
            / "Back"
            / "SiO2"
            / "2D_simulation"
            / "energy_comparison"
            / "simulated_energy_vs_time_copper.csv",
            "output_dir": Path(__file__).resolve().parent / "Copper",
            },
    ]


def load_simulation_front_position(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Front-position CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"time_ns", "front_position_mm", "front_position_cm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    return {
        "time_ns": df["time_ns"].to_numpy(dtype=float),
        "front_position_mm": df["front_position_mm"].to_numpy(dtype=float),
        "front_position_cm": df["front_position_cm"].to_numpy(dtype=float),
    }


def load_model_front_position(wall_name, front_loss_column):
    base_dir = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2" / "1.5 model"
    regular_path = base_dir / "analytic_positions.csv"
    flattop_path = base_dir / "analytic_positions_flattop.csv"

    if not regular_path.exists():
        raise FileNotFoundError(f"Model front-position CSV not found: {regular_path}")
    if not flattop_path.exists():
        raise FileNotFoundError(f"Model front-position CSV not found: {flattop_path}")

    regular = pd.read_csv(regular_path)
    flattop = pd.read_csv(flattop_path)

    marshak_regular = pick_column(regular, "front_position (Marshak)")
    loss_regular = pick_column(regular, front_loss_column)
    marshak_flattop = pick_column(flattop, "front_position (Marshak)")
    loss_flattop = pick_column(flattop, front_loss_column)

    return {
        "time_ns": regular["time_ns"].to_numpy(dtype=float),
        "marshak": regular[marshak_regular].to_numpy(dtype=float),
        "loss_regular": regular[loss_regular].to_numpy(dtype=float),
        "flattop_time_ns": flattop["time_ns"].to_numpy(dtype=float),
        "marshak_flattop": flattop[marshak_flattop].to_numpy(dtype=float),
        "loss_flattop": flattop[loss_flattop].to_numpy(dtype=float),
        "wall_name": wall_name,
    }


def load_simulation_front_surface(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Front-surface CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "r_cm" not in df.columns:
        raise ValueError("CSV must contain 'r_cm' column")
    r_cm = df["r_cm"].to_numpy(dtype=float)
    profiles = {}
    for col in df.columns:
        if col == "r_cm":
            continue
        match = re.match(r"zF_cm_t([\d.]+)ns$", col)
        if not match:
            continue
        profiles[float(match.group(1))] = df[col].to_numpy(dtype=float)
    if not profiles:
        raise ValueError("No front-surface profile columns found.")
    return {"r_cm": r_cm, "profiles": profiles}


def load_model_front_surface(wall_name, front_times_ns=None):
    if front_times_ns is None:
        front_times_ns = FRONT_TIMES_NS

    data_dir = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2" / "2D_shape"
    if not data_dir.exists():
        raise FileNotFoundError(f"Model shape directory not found: {data_dir}")

    regular_profiles = {}
    flattop_profiles = {}
    model_wall = wall_name
    for time_ns in front_times_ns:
        regular_path = data_dir / f"front_shape_{model_wall}_t{time_ns:.2f}ns.csv"
        flattop_path = data_dir / f"front_shape_{model_wall}_flattop_t{time_ns:.2f}ns.csv"

        if not regular_path.exists():
            raise FileNotFoundError(f"Model front-shape CSV not found: {regular_path}")
        if not flattop_path.exists():
            raise FileNotFoundError(f"Model front-shape CSV not found: {flattop_path}")

        regular_df = pd.read_csv(regular_path)
        flattop_df = pd.read_csv(flattop_path)
        required = {"r_cm", "z_F_radial_cm"}
        if not required.issubset(regular_df.columns):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        if not required.issubset(flattop_df.columns):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        regular_profiles[time_ns] = {
            "r_cm": regular_df["r_cm"].to_numpy(dtype=float),
            "z_F_radial_cm": regular_df["z_F_radial_cm"].to_numpy(dtype=float),
        }
        flattop_profiles[time_ns] = {
            "r_cm": flattop_df["r_cm"].to_numpy(dtype=float),
            "z_F_radial_cm": flattop_df["z_F_radial_cm"].to_numpy(dtype=float),
        }

    return {"regular": regular_profiles, "flattop": flattop_profiles}


def load_simulation_energy(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Energy CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"time_ns", "foam_energy_hJ", "gold_energy_hJ"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Energy CSV must contain columns: {sorted(required)}")
    return {
        "time_ns": df["time_ns"].to_numpy(dtype=float),
        "foam_energy_hJ": df["foam_energy_hJ"].to_numpy(dtype=float),
        "coating_energy_hJ": df["gold_energy_hJ"].to_numpy(dtype=float),
    }


def load_model_energy():
    csv_path = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2" / "2D_shape" / "simulated_energy_vs_time_SiO2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Model energy CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"time_ns", "E_marshak", "E_gold_loss", "E_wall_gold_loss", "E_Be_loss", "E_Be_wall_loss", "E_vacuum_loss"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Model energy CSV must contain columns: {sorted(required)}")

    flat_path = Path(BASE_DIR) / "Data_new" / "Back" / "SiO2" / "2D_shape" / "simulated_energy_vs_time_SiO2_flattop.csv"
    if not flat_path.exists():
        raise FileNotFoundError(f"Model energy flattop CSV not found: {flat_path}")
    fdf = pd.read_csv(flat_path)
    missing_flat = required.difference(fdf.columns)
    if missing_flat:
        raise ValueError(f"Model energy flattop CSV must contain columns: {sorted(required)}")

    return {
        "time_ns": df["time_ns"].to_numpy(dtype=float),
        "E_marshak": df["E_marshak"].to_numpy(dtype=float),
        "E_Gold_loss": df["E_gold_loss"].to_numpy(dtype=float),
        "E_Gold_wall_loss": df["E_wall_gold_loss"].to_numpy(dtype=float),
        # "E_Gold_100_loss_flattop": df["E_gold_100_loss"].to_numpy(dtype=float),
        # "E_Gold_100_wall_loss_flattop": df["E_gold_100_wall_loss"].to_numpy(dtype=float),
        "E_Be_loss": df["E_Be_loss"].to_numpy(dtype=float),
        "E_Be_wall_loss": df["E_Be_wall_loss"].to_numpy(dtype=float),
        "E_Cu_loss": df["E_Cu_loss"].to_numpy(dtype=float),
        "E_Cu_wall_loss": df["E_Cu_wall_loss"].to_numpy(np.dtype(float)),
        "E_vacuum_loss": df["E_vacuum_loss"].to_numpy(dtype=float),
        "flattop_time_ns": fdf["time_ns"].to_numpy(dtype=float),
        "E_Gold_loss_flattop": fdf["E_gold_loss"].to_numpy(dtype=float),
        "E_Gold_wall_loss_flattop": fdf["E_wall_gold_loss"].to_numpy(dtype=float),
        # "E_Gold_100_loss_flattop": fdf["E_gold_100_loss"].to_numpy(dtype=float),
        # "E_Gold_100_wall_loss_flattop": fdf["E_gold_100_wall_loss"].to_numpy(dtype=float),
        "E_Be_loss_flattop": fdf["E_Be_loss"].to_numpy(dtype=float),
        "E_Be_wall_loss_flattop": fdf["E_Be_wall_loss"].to_numpy(dtype=float),
        "E_Cu_loss_flattop": fdf["E_Cu_loss"].to_numpy(dtype=float),
        "E_Cu_wall_loss_flattop": fdf["E_Cu_wall_loss"].to_numpy(dtype=float),
        "E_vacuum_loss_flattop": fdf["E_vacuum_loss"].to_numpy(dtype=float),
    }


def plot_front_position(output_dir, wall_name, simulation_data, model_data, loss_label):
    plt.figure(figsize=(8.2, 6.1))
    plt.plot(
        simulation_data["time_ns"],
        simulation_data["front_position_cm"],
        color="blue",
        linestyle="--",
        linewidth=2.8,
        label="Simulation",
    )
    plt.plot(model_data["time_ns"], model_data["marshak"], color="orange", linestyle="-.", linewidth=2.8, label="Model (Marshak)")
    plt.plot(model_data["time_ns"], model_data["loss_regular"], color="firebrick", linestyle="--", linewidth=2.4, label="Model (Henyey)")
    plt.plot(model_data["flattop_time_ns"], model_data["loss_flattop"], color="red", linestyle="-", linewidth=2.6, label="Model (flattop)")
    plt.xlabel(r"$t$ [ns]")
    plt.ylabel(r"$x_F$ [cm]")
    plt.ylim(0.0, 0.2)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = output_dir / "front_position.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def plot_front_surface(output_dir, wall_name, simulation_data, model_data, front_times_ns):
    fig, ax = plt.subplots(figsize=(9.2, 7.8))

    line_styles = {
        "Simulation": ("blue", "--", 2.6),
        "Henyey": ("firebrick", "--", 2.4),
        "flattop": ("red", "-", 2.6),
    }

    for index, time_ns in enumerate(front_times_ns):
        simulation_key = resolve_profile_time_key(simulation_data["profiles"], time_ns)
        sim_color, sim_style, sim_width = line_styles["Simulation"]
        hen_color, hen_style, hen_width = line_styles["Henyey"]
        flat_color, flat_style, flat_width = line_styles["flattop"]

        ax.plot(
            10*simulation_data["r_cm"],
            10*simulation_data["profiles"][simulation_key],
            color=sim_color,
            linestyle=sim_style,
            linewidth=sim_width,
            label=f"Simulation t={time_ns:.1f} ns",
        )
        
        # offset = 0.01 if wall_name == "Be" else 0.0
        # if wall_name == "Vacuum":
        #     offset = 0.002
        # if wall_name == "Gold_100":
        #     offset = 0.008
        z_reg = model_data["regular"][time_ns]["z_F_radial_cm"]# + offset
        z_flat = model_data["flattop"][time_ns]["z_F_radial_cm"]# + offset
        
        ax.plot(
            10*model_data["regular"][time_ns]["r_cm"],
            10*z_reg,
            color=hen_color,
            linestyle=flat_style,
            linewidth=hen_width,
            label=f"Model t={time_ns:.1f} ns",
        )
        # ax.plot(
        #     10*model_data["flattop"][time_ns]["r_cm"],
        #     10*z_flat,
        #     color=flat_color,
        #     linestyle=flat_style,
        #     linewidth=flat_width,
        #     label=f"Model t={time_ns:.1f} ns",
        # )

    ax.set_xlim([0, 10*R_cm])
    ax.set_ylim([0, 10*0.2])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"$r$ [mm]")
    ax.set_ylabel(r"$z_F$ [mm]")
    ax.legend(loc="best", fontsize=14, ncol=1)

    out_path = output_dir / "front_surface.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_energy(output_dir, wall_name, simulation_data, model_data):
    plt.figure(figsize=(8.2, 6.1))
    plt.plot(simulation_data["time_ns"], simulation_data["foam_energy_hJ"], color="blue", linestyle="--", linewidth=2.8, label="Simulation foam energy")
    if wall_name != "Vacuum":
        plt.plot(simulation_data["time_ns"], simulation_data["coating_energy_hJ"], color="royalblue", linestyle="--", linewidth=2.8, label="Simulation coating energy")
    
    # Model - Regular (Henyey)
    plt.plot(model_data["time_ns"], model_data["E_marshak"], color="orange", linestyle="-.", linewidth=2.8, label="Model E (Marshak)")

    if wall_name == "Be":
        # Regular (Henyey)
        plt.plot(model_data["time_ns"], model_data["E_Be_loss"], color="firebrick", linestyle="--", linewidth=2.4, label="Model E Be loss (Henyey)")
        plt.plot(model_data["time_ns"], model_data["E_Be_wall_loss"], color="brown", linestyle="--", linewidth=2.4, label="Model E Be wall loss (Henyey)")
        # Flattop
        plt.plot(model_data["flattop_time_ns"], model_data["E_Be_loss_flattop"], color="red", linestyle="-", linewidth=2.6, label="Model E Be loss (flattop)")
        plt.plot(model_data["flattop_time_ns"], model_data["E_Be_wall_loss_flattop"], color="crimson", linestyle="-", linewidth=2.6, label="Model E Be wall loss (flattop)")
    elif wall_name == "Vacuum":
        plt.plot(model_data["time_ns"], model_data["E_vacuum_loss"], color="firebrick", linestyle="--", linewidth=2.4, label="Model E Vacuum loss (Henyey)")
        # Flattop
        plt.plot(model_data["flattop_time_ns"], model_data["E_vacuum_loss_flattop"], color="red", linestyle="-", linewidth=2.6, label="Model E Vacuum loss (flattop)")
    elif wall_name == "Copper":
        # Regular (Henyey)
        plt.plot(model_data["time_ns"], model_data["E_Cu_loss"], color="firebrick", linestyle="--", linewidth=2.4, label="Model E Copper loss (Henyey)")
        plt.plot(model_data["time_ns"], model_data["E_Cu_wall_loss"], color="brown", linestyle="--", linewidth=2.4, label="Model E Copper wall loss (Henyey)")
        # Flattop
        plt.plot(model_data["flattop_time_ns"], model_data["E_Cu_loss_flattop"], color="red", linestyle="-", linewidth=2.6, label="Model E Copper loss (flattop)")
        plt.plot(model_data["flattop_time_ns"], model_data["E_Cu_wall_loss_flattop"], color="crimson", linestyle="-", linewidth=2.6, label="Model E Copper wall loss (flattop)")
    else:
        # Regular (Henyey)
        plt.plot(model_data["time_ns"], model_data["E_Gold_loss"], color="firebrick", linestyle="--", linewidth=2.4, label="Model E Gold loss (Henyey)")
        plt.plot(model_data["time_ns"], model_data["E_Gold_wall_loss"], color="brown", linestyle="--", linewidth=2.4, label="Model E Gold wall loss (Henyey)")
        # Flattop
        plt.plot(model_data["flattop_time_ns"], model_data["E_Gold_loss_flattop"], color="red", linestyle="-", linewidth=2.6, label="Model E Gold loss (flattop)")
        plt.plot(model_data["flattop_time_ns"], model_data["E_Gold_wall_loss_flattop"], color="crimson", linestyle="-", linewidth=2.6, label="Model E Gold wall loss (flattop)")

    plt.xlabel(r"$t$ [ns]")
    plt.ylabel(r"$E$ [hJ]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = output_dir / "energy.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def run_wall_comparison(config):
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    front_position_sim = load_simulation_front_position(config["simulation_front_position"])
    front_position_model = load_model_front_position(config["name"], config["front_loss_column"])
    front_position_out = plot_front_position(output_dir, config["name"], front_position_sim, front_position_model, config["front_loss_column"])

    front_surface_sim = load_simulation_front_surface(config["simulation_front_surface"])
    available_front_times_ns = sorted(float(key) for key in front_surface_sim["profiles"].keys())
    configured_front_times_ns = config.get("front_times_ns", FRONT_TIMES_NS)
    missing_times = [
        time_ns
        for time_ns in configured_front_times_ns
        if not any(abs(float(time_ns) - available_time) <= 1e-6 for available_time in available_front_times_ns)
    ]
    if missing_times:
        print(
            f"[{config['name']}] configured front times {configured_front_times_ns} do not match "
            f"simulation profile times {available_front_times_ns}; using simulation times."
        )
        front_times_ns = available_front_times_ns
    else:
        front_times_ns = configured_front_times_ns

    front_surface_model = load_model_front_surface(config["name"], front_times_ns)
    front_surface_out = plot_front_surface(output_dir, config["name"], front_surface_sim, front_surface_model, front_times_ns)

    energy_sim = load_simulation_energy(config["simulation_energy"])
    energy_model = load_model_energy()
    energy_out = plot_energy(output_dir, config["name"], energy_sim, energy_model)

    print(f"[{config['name']}] saved: {front_position_out}")
    print(f"[{config['name']}] saved: {front_surface_out}")
    print(f"[{config['name']}] saved: {energy_out}")


def plot_all_coatings_front_surface_at_time(configs, requested_time_ns=COMBINED_TIME_NS):
    common_times = None
    for config in configs:
        simulation_data = load_simulation_front_surface(config["simulation_front_surface"])
        sim_times = {float(key) for key in simulation_data["profiles"].keys()}
        common_times = sim_times if common_times is None else (common_times & sim_times)

    chosen_time_ns = choose_nearest_time(sorted(common_times), requested_time_ns)

    plt.figure(figsize=(9.2, 7.8))
    included = []

    for config in configs:
        wall_name = config["name"]
        if wall_name == "Gold_100":
            continue
        color = COATING_COLORS.get(wall_name, "black")
        try:
            simulation_data = load_simulation_front_surface(config["simulation_front_surface"])
            simulation_key = resolve_profile_time_key(simulation_data["profiles"], chosen_time_ns)
            model_data = load_model_front_surface(wall_name, [chosen_time_ns])

            offset = 0.01 if wall_name == "Be" else 0.0
            if wall_name == "Vacuum":
                offset = 0.002
            if wall_name == "Gold_100":
                offset = 0.008
            if wall_name == "Gold":
                wall_name = "Au"
            elif wall_name == "Copper":
                wall_name = "Cu"
            plt.plot(
                10*simulation_data["r_cm"],
                10*simulation_data["profiles"][simulation_key],
                color=color,
                linestyle="--",
                linewidth=2.4,
                label=f"{wall_name} simulation",
            )
            
            plt.plot(
                10*model_data["regular"][chosen_time_ns]["r_cm"],
                10*model_data["regular"][chosen_time_ns]["z_F_radial_cm"], #+ offset,
                color=color,
                linestyle="-",
                linewidth=2.2,
                label=f"{wall_name} model",
            )
            included.append(wall_name)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[{wall_name}] skipped in combined {chosen_time_ns:.2f} ns plot: {exc}")

    if not included:
        plt.close()
        return None

    plt.xlabel(r"$r$ [mm]")
    plt.ylabel(r"$z_F$ [mm]")
    plt.xlim([0.0, 10*R_cm])
    plt.ylim([0.0, 2])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), fontsize=14, ncol=1, frameon=True)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / f"all_coatings_front_surface_t{chosen_time_ns:g}ns.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    configs = wall_configurations()
    for config in configs:
        run_wall_comparison(config)

    combined_out = plot_all_coatings_front_surface_at_time(configs, COMBINED_TIME_NS)
    if combined_out is not None:
        print(f"[Combined] saved: {combined_out}")


if __name__ == "__main__":
    main()
