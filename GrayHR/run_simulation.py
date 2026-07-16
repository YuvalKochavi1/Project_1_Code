"""
Example and verification script for the Unified Gray HR Model package.

Runs both surface-driven and bath-driven simulations for SiO2 through 4 ns
using the experimental/supplied drive profile, reproducing the numerical 
setup described in Section 8 of the paper.

Saves:
- `front_comparison.png`: Front trajectory vs time for both drives.
- `surface_temperatures.png`: Bath drive vs derived surface radiation and material temperatures.
- `profile_overview.png`: Spatial radiation and material temperature profiles at t = 2.0 ns.
"""

import os
import csv
import numpy as np
from pathlib import Path

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import GrayHRParameters
from solvers import SurfaceDriveSolver, BathDriveSolver


def load_sio2_drive() -> tuple:
    """
    Loads time (ns) and temperature (eV) from T_drive.csv if available.
    Otherwise falls back to a smooth analytic flat-top pulse.
    """
    # Locate T_drive.csv relative to this script
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "Data_new" / "Back" / "SiO2" / "article" / "Temperatures" / "T_drive.csv"
    
    if csv_path.exists():
        t_list, T_list = [], []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 3:
                    t_list.append(float(row[1]))
                    T_list.append(float(row[2]))
        t_arr = np.array(t_list)
        T_arr = np.array(T_list)
        
        # Prepend t=0 if not present
        if t_arr[0] > 0.0:
            t_arr = np.insert(t_arr, 0, 0.0)
            T_arr = np.insert(T_arr, 0, 1.0)
            
        return t_arr, T_arr
    else:
        # Fallback synthetic drive through 4 ns (peaking at ~150 eV)
        t_arr = np.linspace(0.0, 4.0, 200)
        T_arr = 150.0 * (1.0 - np.exp(-t_arr / 0.2)) * np.exp(-max(0.0, t_arr - 2.5) / 1.5)
        return t_arr, T_arr


def main():
    print("====================================================================")
    print(" Running Unified Gray HR Model Verification (SiO2 through 4.0 ns)   ")
    print("====================================================================")
    
    # 1. Load parameters and drive
    params = GrayHRParameters.from_preset("SiO2")
    t_drive, T_drive = load_sio2_drive()
    
    # Ensure evaluation grid up to 4.0 ns
    t_eval = np.linspace(0.0, 4.0, 101)
    
    # 2. Run Surface-Driven Simulation
    print("\n--> [1/2] Running Surface-Driven Solver (T_s prescribed)...")
    surface_solver = SurfaceDriveSolver(params)
    sol_surface = surface_solver.solve(
        times=t_eval,
        T_s_drive=(t_drive, T_drive),
        method="Radau",
        rtol=1e-6,
        atol=1e-8
    )
    print(f"    Done! Front position at 4.0 ns: x_F = {sol_surface.x_F[-1]:.5f} cm")
    
    # 3. Run Bath-Driven Simulation
    print("\n--> [2/2] Running Bath-Driven Solver (T_bath prescribed)...")
    bath_solver = BathDriveSolver(params)
    sol_bath = bath_solver.solve(
        times=t_eval,
        T_bath_drive=(t_drive, T_drive),
        method="Radau",
        rtol=1e-6,
        atol=1e-8
    )
    print(f"    Done! Front position at 4.0 ns: x_F = {sol_bath.x_F[-1]:.5f} cm")
    print(f"          Surface radiation T_s(4.0 ns) = {sol_bath.T_s[-1]:.2f} eV")
    print(f"          Surface material  T_m(4.0 ns) = {sol_bath.T_m[-1]:.2f} eV")
    
    # 4. Print Comparison Table at selected times
    print("\n--------------------------------------------------------------------")
    print(f"{'Time [ns]':<10} | {'x_F (Surface) [cm]':<20} | {'x_F (Bath) [cm]':<20}")
    print("--------------------------------------------------------------------")
    for t_target in [0.5, 1.0, 2.0, 3.0, 4.0]:
        idx = np.argmin(np.abs(t_eval - t_target))
        print(f"{t_eval[idx]:<10.2f} | {sol_surface.x_F[idx]:<20.5f} | {sol_bath.x_F[idx]:<20.5f}")
    print("--------------------------------------------------------------------\n")
    
    # 5. Generate Figures reproducing Section 8
    output_dir = Path(__file__).resolve().parent
    
    # Figure 1: Front Trajectory Comparison
    plt.figure(figsize=(8, 5))
    plt.plot(sol_surface.t, sol_surface.x_F, "b-", label="Surface-driven Gray HR ($T_s$ prescribed)", linewidth=2)
    plt.plot(sol_bath.t, sol_bath.x_F, "r--", label="Bath-driven Gray HR ($T_{\\rm bath}$ prescribed)", linewidth=2)
    plt.plot(sol_surface.t, sol_surface.x_F0, "k:", label="Base Front $x_{F,0}$ (Surface)", linewidth=1.5, alpha=0.7)
    plt.xlabel("Time [ns]", fontsize=12)
    plt.ylabel("Front Position $x_F$ [cm]", fontsize=12)
    plt.title("Surface- and Bath-Driven Gray HR Front Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1_path = output_dir / "front_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig1_path}")
    
    # Figure 2: Bath and Derived Surface Temperatures
    plt.figure(figsize=(8, 5))
    plt.plot(sol_bath.t, [np.interp(ti, t_drive, T_drive) for ti in sol_bath.t], "k-", label="Prescribed Bath $T_{\\rm bath}(t)$", linewidth=2)
    plt.plot(sol_bath.t, sol_bath.T_s, "r-", label="Derived Surface Radiation $T_s(t)$", linewidth=2)
    plt.plot(sol_bath.t, sol_bath.T_m, "b--", label="Derived Surface Material $T_m(t)$", linewidth=2)
    plt.xlabel("Time [ns]", fontsize=12)
    plt.ylabel("Temperature [eV]", fontsize=12)
    plt.title("Bath-Driven Simulation: Prescribed vs Derived Temperatures", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = output_dir / "surface_temperatures.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig2_path}")
    
    # Figure 3: Spatial Profiles overview at t = 2.0 ns
    t_prof = 2.0
    idx_prof = np.argmin(np.abs(sol_surface.t - t_prof))
    xf_prof = sol_surface.x_F[idx_prof]
    x_grid = np.linspace(0.0, xf_prof * 1.1, 200)
    
    tr_prof = sol_surface.get_radiation_profile(t_prof, x_grid)
    tm_prof = sol_surface.get_material_profile(t_prof, x_grid)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid / xf_prof, tr_prof, "r-", label="Radiation Temperature $T_r(x, t)$", linewidth=2)
    plt.plot(x_grid / xf_prof, tm_prof, "b--", label="Material Temperature $T(x, t)$", linewidth=2)
    plt.axvline(1.0, color="k", linestyle=":", alpha=0.6, label="Front Position $x_F$")
    plt.xlabel("Normalized Coordinate $y = x / x_F$", fontsize=12)
    plt.ylabel("Temperature [eV]", fontsize=12)
    plt.title(f"Gray HR Spatial Profiles at $t = {t_prof}$ ns ($s = {sol_surface.s[idx_prof]:.2f}$)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = output_dir / "profile_overview.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig3_path}")
    print("\nVerification completed successfully!")


if __name__ == "__main__":
    main()
