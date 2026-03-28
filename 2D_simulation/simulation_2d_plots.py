from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csv_helpers import ensure_dir, save_series_csv
from simulation_2d_core import K_per_Hev, cell_to_vertices, edges_from_nodes_with_bounds


def _save_temperature_cell_csv(csv_path, r_edges, z_edges, T_cell):
    """Save a 2D temperature field as a flat CSV table with cell-center coordinates."""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    R_cent, Z_cent = np.meshgrid(r_centers, z_centers)

    df = pd.DataFrame(
        {
            "z_cm": Z_cent.ravel(),
            "r_cm": R_cent.ravel(),
            "T_cell_HeV": np.asarray(T_cell).ravel(),
        }
    )
    df.to_csv(csv_path, index=False)


def _save_temperature_vertex_csv(csv_path, r_edges, z_edges, T_vert):
    """Save a vertex-based temperature field table for gouraud shading data."""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    Rv, Zv = np.meshgrid(r_edges, z_edges)
    df = pd.DataFrame(
        {
            "z_cm": Zv.ravel(),
            "r_cm": Rv.ravel(),
            "T_vert_HeV": np.asarray(T_vert).ravel(),
        }
    )
    df.to_csv(csv_path, index=False)


def plot_temperature_maps_gouraud(
    sim,
    stored_t,
    stored_Tm,
    *,
    times_s=(1e-9, 2e-9, 2.5e-9),
    out_dir,
    figure_data_dir,
):
    out_dir = Path(out_dir)
    figure_data_dir = Path(figure_data_dir)
    ensure_dir(out_dir)
    ensure_dir(figure_data_dir / "temperature_maps_gouraud")

    for t_plot in times_s:
        idx_plot = int(np.argmin(np.abs(stored_t - t_plot)))
        T_cell = stored_Tm[idx_plot] / K_per_Hev  # (Nz, Nr) in HeV

        R_total = sim.R_foam + sim.gold_width
        r_edges = edges_from_nodes_with_bounds(sim.r, 0.0, R_total)
        z_edges = edges_from_nodes_with_bounds(sim.z, 0.0, sim.Lz)
        T_vert = cell_to_vertices(T_cell)

        Re, Ze = np.meshgrid(r_edges, z_edges)

        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        pcm = ax.pcolormesh(Re, Ze, T_vert, shading="gouraud", cmap="Spectral_r")

        ax.axvline(sim.R_foam, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("r (cm)", fontname="serif")
        ax.set_ylabel("z (cm)", fontname="serif")
        ax.set_title(f"Material temperature T(r,z) at t = {t_plot*1e9:.1f} ns", fontname="serif")
        ax.set_xlim(0.0, R_total)
        ax.set_ylim(0.0, sim.Lz)

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Material temperature T (HeV)", fontname="serif")

        time_ns = t_plot * 1e9
        fig_name = f"Tmap_{time_ns:.1f}ns"
        fig.savefig(out_dir / f"{fig_name}.png", dpi=250)
        plt.close(fig)

        _save_temperature_cell_csv(
            figure_data_dir / "temperature_maps_gouraud" / f"{fig_name}.csv",
            r_edges,
            z_edges,
            T_cell,
        )
        _save_temperature_vertex_csv(
            figure_data_dir / "temperature_maps_gouraud" / f"{fig_name}_vertex.csv",
            r_edges,
            z_edges,
            T_vert,
        )


def plot_temperature_maps_simple(
    sim,
    stored_t,
    stored_Tm,
    *,
    times_s=(1e-9, 2e-9, 2.5e-9),
    out_dir,
    figure_data_dir,
):
    out_dir = Path(out_dir)
    figure_data_dir = Path(figure_data_dir)
    ensure_dir(out_dir)
    ensure_dir(figure_data_dir / "temperature_maps_simple")

    for t_plot in times_s:
        idx_plot = int(np.argmin(np.abs(stored_t - t_plot)))
        R_total = sim.R_foam + sim.gold_width
        r_edges = edges_from_nodes_with_bounds(sim.r, 0.0, R_total)
        z_edges = edges_from_nodes_with_bounds(sim.z, 0.0, sim.Lz)
        Tm = stored_Tm[idx_plot]

        plt.figure()
        plt.pcolormesh(r_edges, z_edges, Tm, shading="auto", cmap="Spectral_r")
        plt.axvline(sim.R_foam, linestyle="--")
        plt.xlim(0.0, R_total)
        plt.ylim(0.0, sim.Lz)
        plt.colorbar(label="Material Temperature")
        plt.title(f"Material Temperature Tm at t={t_plot*1e9:.1f} ns")
        plt.xlabel("r (cm)")
        plt.ylabel("z (cm)")

        time_ns = t_plot * 1e9
        fig_name = f"heatmap_{time_ns:.1f}ns"
        plt.savefig(out_dir / f"{fig_name}.png")
        plt.close()

        _save_temperature_cell_csv(
            figure_data_dir / "temperature_maps_simple" / f"{fig_name}.csv",
            r_edges,
            z_edges,
            Tm,
        )


def plot_front_vs_time(
    stored_t,
    front_z_cm,
    *,
    out_path,
    figure_data_dir,
    base_dir,
):
    out_path = Path(out_path)
    base_dir = Path(base_dir)
    figure_data_dir = Path(figure_data_dir)
    ensure_dir(out_path.parent)

    plt.figure(figsize=(8, 6))
    plt.plot(stored_t * 1e9, np.asarray(front_z_cm) * 1e1)

    overlays = [
        (figure_data_dir / "gold_supersonic_comparison" / "shay_model_first_sent.csv", "Gold wall - Supersonic (shay's model)", "--", "blue", 1.0),
        (figure_data_dir / "gold_supersonic_comparison" / "shay_simulation.csv", "Gold wall - Supersonic (Avner's simulation)", "-", "blue", 1.0),
        (figure_data_dir / "gold_supersonic_comparison" / "gold_wall_subsonic.csv", "Gold wall - Subsonic (my 1.5 model)", "-", "red", 1.0),
        (figure_data_dir / "gold_supersonic_comparison" / "my1_5_model_supersonic.csv", "Gold wall - Supersonic (my 1.5 model)", "--", "red", 10.0),
    ]

    for path, label, ls, color, yscale in overlays:
        if path.exists():
            df = pd.read_csv(path)
            t_csv = df["x"].to_numpy()
            x_csv = df["y"].to_numpy()
            plt.plot(t_csv, yscale * x_csv, linestyle=ls, label=label, color=color)

    plt.xlabel("Time (ns)", fontname="serif")
    plt.ylabel("Front Position (millimeters)", fontname="serif")
    plt.title("Front Position vs Time at r=0", fontname="serif")
    plt.ylim(0, 2)
    plt.grid()
    plt.legend(prop={"family": "serif"})
    plt.savefig(out_path)
    plt.close()

    ensure_dir(figure_data_dir / "front_vs_time")
    save_series_csv(
        figure_data_dir / "front_vs_time" / "front_position_vs_time_r0.csv",
        {
            "time_ns": np.asarray(stored_t) * 1e9,
            "front_position_mm": np.asarray(front_z_cm) * 1e1,
            "front_position_cm": np.asarray(front_z_cm),
        },
    )


def plot_front_surface(
    sim,
    stored_t,
    stored_Tm,
    *,
    times_s=(1e-9, 2e-9, 2.5e-9),
    out_path,
    figure_data_dir,
):
    out_path = Path(out_path)
    figure_data_dir = Path(figure_data_dir)
    ensure_dir(out_path.parent)
    R_total = sim.R_foam + sim.gold_width

    zF = sim.compute_front_surface(stored_Tm, front_method="maxgrad")
    plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    for t_plot in times_s:
        idx_plot = int(np.argmin(np.abs(stored_t - t_plot)))
        plt.plot(sim.r, zF[idx_plot], label=f"t={t_plot*1e9:.1f} ns")

    plt.axvline(sim.R_foam, color="k", linestyle="--", linewidth=1, label="Foam-Gold Interface")
    plt.xlabel("r (cm)", fontname="serif")
    plt.ylabel("Front Position z_F (cm)", fontname="serif")
    plt.title("Front Surface z_F(r) at Different Times", fontname="serif")
    plt.xlim(0.0, R_total)
    plt.ylim(0.0, sim.Lz)
    plt.grid()
    plt.legend(prop={"family": "serif"})
    plt.savefig(out_path)
    plt.close()

    ensure_dir(figure_data_dir / "front_surface")
    export_dict = {"r_cm": sim.r}
    for t_plot in times_s:
        idx_plot = int(np.argmin(np.abs(stored_t - t_plot)))
        export_dict[f"zF_cm_t{t_plot*1e9:.1f}ns"] = zF[idx_plot]
    save_series_csv(figure_data_dir / "front_surface" / "front_surface_profiles.csv", export_dict)


def plot_energy_comparison(
    sim,
    stored_t,
    stored_Um,
    *,
    out_path,
    figure_data_dir,
    base_dir,
    material,
    experiment,
):
    out_path = Path(out_path)
    base_dir = Path(base_dir)
    figure_data_dir = Path(figure_data_dir)
    ensure_dir(out_path.parent)

    energy_foam = sim.compute_energy_foam(stored_Um) * 1e-9
    energy_gold = sim.compute_energy_gold(stored_Um) * 1e-9

    plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    plt.plot(stored_t * 1e9, energy_foam, label="Simulated Foam Energy (hJ)", color="blue")
    plt.plot(stored_t * 1e9, energy_gold, label="Simulated Gold Energy (hJ)", color="green")

    overlay_paths = [
        (figure_data_dir / "gold_supersonic_comparison" / "supersonic_gold_lost_energy.csv", "Estimated Lost Energy", "--", "red"),
        (base_dir/ "Data_new" / experiment / material / "article" / "energies" / "total_energy_1D.csv", "Total Energy 1D", "-.", "orange"),
        (base_dir / "Data_new" / experiment / material / "article" / "energies" / "total_energy_2D.csv", "Total Energy 2D", "--", "purple"),
    ]

    for path, label, ls, color in overlay_paths:
        if path.exists():
            df = pd.read_csv(path)
            t_csv = df["x"].to_numpy()
            energy_csv = df["y"].to_numpy()
            plt.plot(t_csv, energy_csv, linestyle=ls, label=label, color=color)

    plt.xlabel("Time (ns)", fontname="serif")
    plt.ylabel("Foam Energy (hJ)", fontname="serif")
    plt.title("Foam Energy vs Time", fontname="serif")
    plt.grid()
    plt.legend(prop={"family": "serif"})
    plt.savefig(out_path)
    plt.close()

    ensure_dir(figure_data_dir / "energy_comparison")
    save_series_csv(
        figure_data_dir / "energy_comparison" / "simulated_energy_vs_time.csv",
        {
            "time_ns": np.asarray(stored_t) * 1e9,
            "foam_energy_hJ": energy_foam,
            "gold_energy_hJ": energy_gold,
        },
    )
