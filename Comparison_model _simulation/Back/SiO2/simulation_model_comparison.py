"""SiO2 (Back) — Gold-only comparison plots.

Generates per-wall figures under a local `Gold` output folder. Placeholders
for `Be` and `Copper` are left commented for future extension.
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

from parameters import Experiment, R_cm, L, Material
from model_main import BASE_DIR

print(f"Experiment: {Experiment}, Material: {Material}")

FRONT_TIMES_NS = [1.0, 2.0, 2.5]


def pick_column(df, *wanted_names):
    normalized = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    for w in wanted_names:
        k = re.sub(r"[^a-z0-9]+", "", w.lower())
        if k in normalized:
            return normalized[k]
    raise KeyError(f"None of columns found: {wanted_names}")


def output_folder_for(wall):
    out = Path(__file__).resolve().parent / wall
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_sim_front_surface(csv_path):
    df = pd.read_csv(csv_path)
    if "r_cm" not in df.columns:
        raise ValueError("CSV must contain r_cm")
    r = df["r_cm"].to_numpy(dtype=float)
    profiles = {}
    for col in df.columns:
        if col == "r_cm":
            continue
        m = re.match(r"zF_cm_t([\d.]+)ns$", col)
        if not m:
            continue
        profiles[float(m.group(1))] = df[col].to_numpy(dtype=float)
    return {"r_cm": r, "profiles": profiles}


def load_model_front_shapes(wall):
    data_dir = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape"
    regular = {}
    flattop = {}
    for t in FRONT_TIMES_NS:
        reg = data_dir / f"front_shape_{wall}_t{t:.2f}ns.csv"
        flat = data_dir / f"front_shape_{wall}_flattop_t{t:.2f}ns.csv"
        if not reg.exists() or not flat.exists():
            raise FileNotFoundError(f"Missing model front files for {wall} at t={t}")
        rdf = pd.read_csv(reg); fdf = pd.read_csv(flat)
        regular[t] = {"r_cm": rdf["r_cm"].to_numpy(), "z_F_radial_cm": rdf["z_F_radial_cm"].to_numpy()}
        flattop[t] = {"r_cm": fdf["r_cm"].to_numpy(), "z_F_radial_cm": fdf["z_F_radial_cm"].to_numpy()}
    return {"regular": regular, "flattop": flattop}


def load_sim_front_position(csv_path):
    df = pd.read_csv(csv_path)
    return {"time_ns": df["time_ns"].to_numpy(), "front_position_cm": df["front_position_cm"].to_numpy()}


def load_model_front_positions():
    base = Path(BASE_DIR) / "Data_new" / Experiment / Material / "1.5 model"
    reg = pd.read_csv(base / "analytic_positions.csv")
    flat = pd.read_csv(base / "analytic_positions_flattop.csv")
    # Expect columns like front_position (Marshak), front_position (Gold Loss)
    marshak = pick_column(reg, "front_position (Marshak)")
    gold_loss = pick_column(reg, "front_position (Gold Loss)")
    gold_loss_flat = pick_column(flat, "front_position (Gold Loss)")
    return {
        "time_ns": reg["time_ns"].to_numpy(),
        "marshak": reg[marshak].to_numpy(),
        "gold_loss": reg[gold_loss].to_numpy(),
        "flattop_time_ns": flat["time_ns"].to_numpy(),
        "gold_loss_flattop": flat[gold_loss_flat].to_numpy(),
    }


def load_sim_energy(csv_path):
    df = pd.read_csv(csv_path)
    return {"time_ns": df["time_ns"].to_numpy(), "foam_energy_hJ": df["foam_energy_hJ"].to_numpy(), "coating_energy_hJ": df["gold_energy_hJ"].to_numpy()}


def load_model_energy():
    csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape" / "simulated_energy_vs_time_SiO2.csv"
    if not csv_path.exists():
        # fallback to generic name
        csv_path = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape" / "simulated_energy_vs_time.csv"
    df = pd.read_csv(csv_path)
    return {
        "time_ns": df["time_ns"].to_numpy(),
        "E_marshak": df["E_marshak"].to_numpy(),
        "E_Gold_loss": df.get("E_Gold_loss", df.get("E_gold_loss", np.zeros_like(df["time_ns"]))),
        "E_Gold_wall_loss": df.get("E_Gold_wall_loss", np.zeros_like(df["time_ns"])),
    }


def plot_front_surface_panels(output_dir, wall, sim, model):
    colors = {1.0: "tab:blue", 2.0: "tab:orange", 2.5: "tab:green"}
    fig, axes = plt.subplots(1, len(FRONT_TIMES_NS), figsize=(5.0 * len(FRONT_TIMES_NS), 4.6), sharey=True)
    if len(FRONT_TIMES_NS) == 1:
        axes = [axes]
    for i, t in enumerate(FRONT_TIMES_NS):
        ax = axes[i]
        c = colors[t]
        ax.plot(sim["r_cm"], sim["profiles"][t], color=c, linestyle="--", linewidth=2.2, label=f"Simulation t={t:.1f} ns")
        ax.plot(model["regular"][t]["r_cm"], model["regular"][t]["z_F_radial_cm"], color=c, linewidth=2.2, label=f"Model heyney {t:.1f} ns")
        ax.plot(model["flattop"][t]["r_cm"], model["flattop"][t]["z_F_radial_cm"], color=c, linestyle=":", linewidth=2.4, label=f"Model flattop {t:.1f} ns")
        ax.set_xlim([0, R_cm])
        ax.set_ylim([0, L])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Model {t:.1f} ns")
        ax.set_xlabel("r (cm)")
        if i == 0:
            ax.set_ylabel("z_F (cm)")
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Simulation vs Model Front Comparison at Each Time\n({Experiment} - {Material} - {wall})", fontsize=13)
    fig.tight_layout()
    out = output_dir / "front_surface.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_front_position(output_dir, wall, sim, model):
    plt.figure(figsize=(8, 6))
    plt.plot(sim["time_ns"], sim["front_position_cm"], color="tab:blue", linestyle="--", linewidth=2.2, label="Simulation front (r=0)")
    plt.plot(model["time_ns"], model["marshak"], color="tab:red", linewidth=2.3, label="Model (Marshak)")
    plt.plot(model["time_ns"], model["gold_loss"], color="tab:green", linewidth=2.2, label="Model (Gold Loss) - heyney")
    plt.plot(model["flattop_time_ns"], model["gold_loss_flattop"], color="tab:purple", linestyle=":", linewidth=2.4, label="Model (Gold Loss) - flattop")
    plt.xlabel("Time (ns)")
    plt.ylabel("Front position (cm)")
    plt.title(f"Front Position vs Time ({Experiment} - {Material} - {wall})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    out = output_dir / "front_position.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def plot_energy(output_dir, wall, sim, model):
    plt.figure(figsize=(8, 6))
    plt.plot(sim["time_ns"], sim["foam_energy_hJ"], color="tab:blue", linestyle="--", linewidth=2.2, label="Simulation foam energy")
    plt.plot(sim["time_ns"], sim["coating_energy_hJ"], color="tab:orange", linestyle="-.", linewidth=2.2, label="Simulation coating energy")
    plt.plot(model["time_ns"], model["E_marshak"], color="tab:red", linewidth=2.3, label="Model E_marshak")
    plt.plot(model["time_ns"], model["E_Gold_loss"], color="tab:green", linewidth=2.2, label="Model E_Gold_loss")
    plt.plot(model["time_ns"], model["E_Gold_wall_loss"], color="tab:purple", linestyle=":", linewidth=2.3, label="Model E_Gold_wall_loss")
    plt.xlabel("Time (ns)")
    plt.ylabel("Energy (hJ)")
    plt.title(f"Energy Comparison ({Experiment} - {Material} - {wall})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    out = output_dir / "energy.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def main():
    wall = "Gold"
    out = output_folder_for(wall)

    sim_front_surface = load_sim_front_surface(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_surface" / "front_surface_profiles300.csv")
    model_shapes = load_model_front_shapes("Gold")
    psurf = plot_front_surface_panels(out, wall, sim_front_surface, model_shapes)

    sim_front_pos = load_sim_front_position(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_vs_time" / "front_position_vs_time_r0300.csv")
    model_pos = load_model_front_positions()
    ppos = plot_front_position(out, wall, sim_front_pos, model_pos)

    sim_energy = load_sim_energy(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "energy_comparison" / "simulated_energy_vs_time300.csv")
    model_energy = load_model_energy()
    peng = plot_energy(out, wall, sim_energy, model_energy)

    print(f"Saved: {psurf}, {ppos}, {peng}")


if __name__ == "__main__":
    main()

# Future walls (placeholders):
# Be config: use front_surface_profiles_be.csv, front_position_vs_time_Be_r0.csv, simulated_energy_vs_time_be.csv
# Copper config: use front_surface_profiles_copper.csv, front_position_vs_time_Copper_r0.csv, simulated_energy_vs_time_copper.csv

# ---------------------------------------------------------------------------
# Copper variant (commented):
#
# def main_copper():
#     wall = "Copper"
#     out = output_folder_for(wall)
#
#     sim_front_surface = load_sim_front_surface(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_surface" / "front_surface_profiles_copper.csv")
#     model_shapes = load_model_front_shapes("Copper")
#     psurf = plot_front_surface_panels(out, wall, sim_front_surface, model_shapes)
#
#     sim_front_pos = load_sim_front_position(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_vs_time" / "front_position_vs_time_Copper_r0.csv")
#     model_pos = load_model_front_positions()
#     ppos = plot_front_position(out, wall, sim_front_pos, model_pos)
#
#     # Energy CSV may use column naming like 'copper_energy_hJ' and model CSV may have E_Copper_loss
#     sim_energy = pd.read_csv(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "energy_comparison" / "simulated_energy_vs_time_copper.csv")
#     sim_energy_dict = {"time_ns": sim_energy["time_ns"].to_numpy(), "foam_energy_hJ": sim_energy["foam_energy_hJ"].to_numpy(), "coating_energy_hJ": sim_energy["copper_energy_hJ"].to_numpy()}
#     model_energy_df = pd.read_csv(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape" / "simulated_energy_vs_time_SiO2.csv")
#     model_energy_dict = {
#         "time_ns": model_energy_df["time_ns"].to_numpy(),
#         "E_marshak": model_energy_df["E_marshak"].to_numpy(),
#         "E_Copper_loss": model_energy_df.get("E_Copper_loss", np.zeros_like(model_energy_df["time_ns"])),
#         "E_Copper_wall_loss": model_energy_df.get("E_Copper_wall_loss", np.zeros_like(model_energy_df["time_ns"])),
#     }
#     peng = plot_energy(out, wall, sim_energy_dict, model_energy_dict)
#
#     print(f"Saved (Copper): {psurf}, {ppos}, {peng}")
#
# ---------------------------------------------------------------------------
# Beryllium (Be) variant (commented):
#
# def main_be():
#     wall = "Be"
#     out = output_folder_for(wall)
#
#     sim_front_surface = load_sim_front_surface(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_surface" / "front_surface_profiles_be.csv")
#     model_shapes = load_model_front_shapes("Be")
#     psurf = plot_front_surface_panels(out, wall, sim_front_surface, model_shapes)
#
#     sim_front_pos = load_sim_front_position(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "front_vs_time" / "front_position_vs_time_Be_r0.csv")
#     model_pos = load_model_front_positions()
#     ppos = plot_front_position(out, wall, sim_front_pos, model_pos)
#
#     # Energy CSV may use column naming like 'be_energy_hJ' and model CSV may have E_Be_loss
#     sim_energy = pd.read_csv(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_simulation" / "energy_comparison" / "simulated_energy_vs_time_be.csv")
#     sim_energy_dict = {"time_ns": sim_energy["time_ns"].to_numpy(), "foam_energy_hJ": sim_energy["foam_energy_hJ"].to_numpy(), "coating_energy_hJ": sim_energy["be_energy_hJ"].to_numpy()}
#     model_energy_df = pd.read_csv(Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape" / "simulated_energy_vs_time_SiO2.csv")
#     model_energy_dict = {
#         "time_ns": model_energy_df["time_ns"].to_numpy(),
#         "E_marshak": model_energy_df["E_marshak"].to_numpy(),
#         "E_Be_loss": model_energy_df.get("E_Be_loss", np.zeros_like(model_energy_df["time_ns"])),
#         "E_Be_wall_loss": model_energy_df.get("E_Be_wall_loss", np.zeros_like(model_energy_df["time_ns"])),
#     }
#     peng = plot_energy(out, wall, sim_energy_dict, model_energy_dict)
#
#     print(f"Saved (Be): {psurf}, {ppos}, {peng}")
#
