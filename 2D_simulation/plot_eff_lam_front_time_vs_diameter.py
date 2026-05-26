"""Plot the time for the foam front to reach 0.75 mm versus foam diameter.

This script scans the eff_lam run folders produced by
simulation_2d_pipeline.run_eff_lam_radius_sweep(...), reads each
front_position_vs_time_r0.csv file, interpolates the first time at which the
foam front reaches 0.75 mm, and plots that time as a function of the foam
diameter.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csv_helpers import ensure_dir

BASE_DIR = PROJECT_ROOT
DATA_ROOT = BASE_DIR / "Data_new" / "Back" / "SiO2" / "2D_simulation" / "eff_lam"
ANALYTIC_ROOT = BASE_DIR / "Data_new" / "Back" / "SiO2" / "2D_shape" / "eff_lam"
FIGURE_ROOT = BASE_DIR / "Figures_new" / "Back" / "SiO2" / "2D_simulation" / "eff_lam"
TARGET_FRONT_MM = 1


def _radius_from_folder_name(folder_name: str) -> float:
    """Parse a folder name like `R_0p01` or `R_0p01_g1` into a radius in cm."""
    match = re.search(r"R_(\d+p\d+|\d+)", folder_name)
    if not match:
        raise ValueError(f"Cannot parse radius from folder name: {folder_name}")
    return float(match.group(1).replace("p", "."))


def _load_crossing_time(csv_path: Path, target_front_mm: float = TARGET_FRONT_MM) -> float:
    """Return the first time in ns when the front reaches target_front_mm.

    Uses linear interpolation between the two nearest samples that straddle the
    target front position. If the front never reaches the target, returns NaN.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return float("nan")

    time_ns = np.asarray(data["time_ns"], dtype=float)
    front_mm = np.asarray(data["front_position_mm"], dtype=float)

    valid = np.isfinite(time_ns) & np.isfinite(front_mm)
    time_ns = time_ns[valid]
    front_mm = front_mm[valid]

    if time_ns.size == 0:
        return float("nan")

    reached = np.where(front_mm >= target_front_mm)[0]
    if reached.size == 0:
        return float("nan")

    idx = int(reached[0])
    if idx == 0:
        return float(time_ns[0])

    t0, t1 = float(time_ns[idx - 1]), float(time_ns[idx])
    f0, f1 = float(front_mm[idx - 1]), float(front_mm[idx])
    if f1 == f0:
        return t1

    frac = (target_front_mm - f0) / (f1 - f0)
    return t0 + frac * (t1 - t0)


def gather_radius_sweep_results(data_root: Path = DATA_ROOT, g_val: str | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Collect diameters and crossing times from the eff_lam sweep folders."""
    rows: list[tuple[float, float]] = []

    for folder_dir in sorted(data_root.glob("R_*")):
        csv_files = list(folder_dir.glob("front_vs_time/front_position_vs_time*r0.csv"))
        if not csv_files:
            continue
        folder = csv_files[0]
        radius_folder = folder_dir.name
        
        folder_suffix = ""
        if "_g1e-06" in radius_folder:
            folder_suffix = "g1e-06"
        elif "_g1" in radius_folder:
            folder_suffix = "g1"

        target_suffix = ""
        if g_val == 1 or str(g_val) == "1" or g_val == "g1":
            target_suffix = "g1"
        elif g_val == "1e-06" or g_val == "g1e-06":
            target_suffix = "g1e-06"
            
        if folder_suffix != target_suffix:
            continue

        radius_cm = _radius_from_folder_name(radius_folder)
        diameter_mm = 2.0 * radius_cm * 10.0
        crossing_time_ns = _load_crossing_time(folder, TARGET_FRONT_MM)
        rows.append((diameter_mm, crossing_time_ns))

    if not rows:
        return np.array([]), np.array([])

    rows.sort(key=lambda item: item[0])
    diameters_mm = np.array([item[0] for item in rows], dtype=float)
    crossing_times_ns = np.array([item[1] for item in rows], dtype=float)
    return diameters_mm, crossing_times_ns


def gather_analytic_front_sweep_results(data_root: Path = ANALYTIC_ROOT, column: str = "front_position_gold_loss_cm") -> tuple[np.ndarray, np.ndarray]:
    """Collect diameters and crossing times from the analytic sweep folders."""
    rows: list[tuple[float, float]] = []

    for folder_dir in sorted(data_root.glob("R_*")):
        csv_files = list(folder_dir.glob("front_vs_time/analytic_positions.csv"))
        if not csv_files:
            continue
        folder = csv_files[0]
        radius_folder = folder_dir.name

        radius_cm = _radius_from_folder_name(radius_folder)
        diameter_mm = 2.0 * radius_cm * 10.0
        
        # calculate crossing
        data = np.genfromtxt(folder, delimiter=",", names=True)
        if data.size == 0:
            continue

        time_ns = np.asarray(data["time_ns"], dtype=float)
        front_cm = np.asarray(data[column], dtype=float)
        front_mm = front_cm * 10.0

        valid = np.isfinite(time_ns) & np.isfinite(front_mm)
        time_ns = time_ns[valid]
        front_mm = front_mm[valid]

        reached = np.where(front_mm >= TARGET_FRONT_MM)[0]
        if reached.size == 0:
            crossing_time_ns = float("nan")
        else:
            idx = int(reached[0])
            if idx == 0:
                crossing_time_ns = float(time_ns[0])
            else:
                t0, t1 = float(time_ns[idx - 1]), float(time_ns[idx])
                f0, f1 = float(front_mm[idx - 1]), float(front_mm[idx])
                if f1 == f0:
                    crossing_time_ns = t1
                else:
                    frac = (TARGET_FRONT_MM - f0) / (f1 - f0)
                    crossing_time_ns = t0 + frac * (t1 - t0)

        rows.append((diameter_mm, crossing_time_ns))

    if not rows:
        return np.array([]), np.array([])

    rows.sort(key=lambda item: item[0])
    diameters_mm = np.array([item[0] for item in rows], dtype=float)
    crossing_times_ns = np.array([item[1] for item in rows], dtype=float)
    return diameters_mm, crossing_times_ns


def plot_front_time_vs_diameter(*, data_root: Path = DATA_ROOT, figure_root: Path = FIGURE_ROOT) -> Path:
    """Create and save the diameter vs. time plot."""
    diameters_mm, crossing_times_ns = gather_radius_sweep_results(data_root)
    diameters_mm_g1, crossing_times_ns_g1 = gather_radius_sweep_results(data_root, g_val=1)
    diameters_mm_g1e_06, crossing_times_ns_g1e_06 = gather_radius_sweep_results(data_root, g_val="g1e-06")

    diameters_mm_ana_loss, crossing_times_ana_loss = gather_analytic_front_sweep_results(ANALYTIC_ROOT, "front_position_gold_loss_cm")
    diameters_mm_ana_lam_eff, crossing_times_ana_lam_eff = gather_analytic_front_sweep_results(ANALYTIC_ROOT, "front_position_gold_loss_lam_eff_cm")
    diameters_mm_ana_lam_eff_p1, crossing_times_ana_lam_eff_p1 = gather_analytic_front_sweep_results(ANALYTIC_ROOT, "front_position_gold_loss_lam_eff_power_1_cm")
    diameters_mm_ana_lam_eff_p2, crossing_times_ana_lam_eff_p2 = gather_analytic_front_sweep_results(ANALYTIC_ROOT, "front_position_gold_loss_lam_eff_power_2_cm")
    ensure_dir(figure_root)
    figure_path = figure_root / "time_to_front_0p75mm_vs_diameter.png"

    # use a serif font for all text in the figure
    plt.rcParams["font.family"] = "serif"
    fig, main_ax = plt.subplots(figsize=(8, 5))
    
    # main_ax.plot(diameters_mm, crossing_times_ns, marker="o", linewidth=2.0, color="tab:blue", label="x100 opaque Gold")
    if len(diameters_mm_g1) > 0:
        main_ax.plot(diameters_mm_g1, crossing_times_ns_g1, marker="s", linewidth=2.0, color="tab:green", label="Nominal Gold (Sim)")
    # if len(diameters_mm_g1e_06) > 0:
    #     main_ax.plot(diameters_mm_g1e_06, crossing_times_ns_g1e_06, marker="^", linewidth=2.0, color="tab:red", label="g = 1e-06")

    if len(diameters_mm_ana_loss) > 0:
        main_ax.plot(diameters_mm_ana_loss, crossing_times_ana_loss, marker="x", linewidth=2.0, color="tab:purple", label="Analytic (Gold loss)")
    if len(diameters_mm_ana_lam_eff) > 0:
        main_ax.plot(diameters_mm_ana_lam_eff, crossing_times_ana_lam_eff, marker="d", linewidth=2.0, color="tab:brown", label="Analytic (Gold loss + lam_eff)")
    if len(diameters_mm_ana_lam_eff_p1) > 0:
        main_ax.plot(diameters_mm_ana_lam_eff_p1, crossing_times_ana_lam_eff_p1, marker="v", linewidth=2.0, color="tab:cyan", label="Analytic (Gold loss + lam_eff_1)")
    if len(diameters_mm_ana_lam_eff_p2) > 0:
        main_ax.plot(diameters_mm_ana_lam_eff_p2, crossing_times_ana_lam_eff_p2, marker="^", linewidth=2.0, color="tab:pink", label="Analytic (Gold loss + lam_eff_2)")
    # add a y line at y= 1.057 and call it mmarshak_1D result
    main_ax.axhline(0.773, color="tab:orange", linestyle="--", label="Marshak 1D")
    main_ax.set_xlabel("Diameter (mm)")
    main_ax.set_ylabel(f"Time for front to reach {TARGET_FRONT_MM} mm (ns)")
    main_ax.grid(True, alpha=0.3)
    main_ax.legend()

    # Create inset axis
    ins_ax = main_ax.inset_axes([0.05, 0.05, 0.3, 0.3])
    # ins_ax.plot(diameters_mm, crossing_times_ns, marker="o", linewidth=2.0, color="tab:blue")
    if len(diameters_mm_g1) > 0:
        ins_ax.plot(diameters_mm_g1, crossing_times_ns_g1, marker="s", linewidth=2.0, color="tab:green")
    # if len(diameters_mm_g1e_06) > 0:
    #     ins_ax.plot(diameters_mm_g1e_06, crossing_times_ns_g1e_06, marker="^", linewidth=2.0, color="tab:red")
    
    if len(diameters_mm_ana_loss) > 0:
        ins_ax.plot(diameters_mm_ana_loss, crossing_times_ana_loss, marker="x", linewidth=2.0, color="tab:purple")
    if len(diameters_mm_ana_lam_eff) > 0:
        ins_ax.plot(diameters_mm_ana_lam_eff, crossing_times_ana_lam_eff, marker="d", linewidth=2.0, color="tab:brown")
    if len(diameters_mm_ana_lam_eff_p1) > 0:
        ins_ax.plot(diameters_mm_ana_lam_eff_p1, crossing_times_ana_lam_eff_p1, marker="v", linewidth=2.0, color="tab:cyan")
    if len(diameters_mm_ana_lam_eff_p2) > 0:
        ins_ax.plot(diameters_mm_ana_lam_eff_p2, crossing_times_ana_lam_eff_p2, marker="^", linewidth=2.0, color="tab:pink")

    ins_ax.axhline(0.773, color="tab:orange", linestyle="--")
    ins_ax.set_xscale("log")
    ins_ax.set_xlim(5e-3, 2)
    ins_ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {figure_path}")
    print("Original: Diameter_mm, time_ns")
    for d_mm, t_ns in zip(diameters_mm, crossing_times_ns):
        print(f"{d_mm:.6g}, {t_ns:.6g}")
    print("g=1: Diameter_mm, time_ns")
    for d_mm, t_ns in zip(diameters_mm_g1, crossing_times_ns_g1):
        print(f"{d_mm:.6g}, {t_ns:.6g}")
    print("g=1e-06: Diameter_mm, time_ns")
    for d_mm, t_ns in zip(diameters_mm_g1e_06, crossing_times_ns_g1e_06):
        print(f"{d_mm:.6g}, {t_ns:.6g}")

    return figure_path


def gather_analytic_energy_sweep_results(data_root: Path = ANALYTIC_ROOT, column_gold: str = "E_wall_gold_loss_hJ", column_foam: str = "E_foam_gold_loss_hJ") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect diameters and final energies from the analytic sweep folders."""
    rows: list[tuple[float, float, float]] = []

    for folder_dir in sorted(data_root.glob("R_*")):
        csv_files = list(folder_dir.glob("energy_comparison/analytic_energy_vs_time.csv"))
        if not csv_files:
            continue
        folder = csv_files[0]
        radius_folder = folder_dir.name

        radius_cm = _radius_from_folder_name(radius_folder)
        diameter_mm = 2.0 * radius_cm * 10.0
        
        try:
            data = np.genfromtxt(folder, delimiter=",", names=True)
            if data.size == 0:
                continue
            
            gold_energy = np.asarray(data[column_gold], dtype=float)
            foam_energy = np.asarray(data[column_foam], dtype=float)
            
            valid = np.isfinite(gold_energy)
            if not np.any(valid):
                continue
            
            final_gold = float(gold_energy[valid][-1])
            final_foam = float(foam_energy[valid][-1])
            
            rows.append((diameter_mm, final_gold, final_foam))
        except Exception:
            continue

    if not rows:
        return np.array([]), np.array([]), np.array([])

    rows.sort(key=lambda item: item[0])
    diameters_mm = np.array([item[0] for item in rows], dtype=float)
    gold_energies_hJ = np.array([item[1] for item in rows], dtype=float)
    foam_energies_hJ = np.array([item[2] for item in rows], dtype=float)

    return diameters_mm, gold_energies_hJ, foam_energies_hJ

def _load_final_gold_foam_energy(csv_path: Path) -> tuple[float, float]:
    """Return the gold and foam energies at the final time from the simulation csv."""
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        if data.size == 0:
            return float("nan"), float("nan")
        
        gold_energy = np.asarray(data["gold_energy_hJ"], dtype=float)
        foam_energy = np.asarray(data["foam_energy_hJ"], dtype=float)
        valid = np.isfinite(gold_energy)
        if not np.any(valid):
            return float("nan"), float("nan")
        
        return float(gold_energy[valid][-1]), float(foam_energy[valid][-1])
    except Exception:
        return float("nan"), float("nan")


def gather_energy_sweep_results(data_root: Path = DATA_ROOT, g_val: str | int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect diameters and final gold and foam energies from the eff_lam sweep folders."""
    rows: list[tuple[float, float, float]] = []

    for folder_dir in sorted(data_root.glob("R_*")):
        csv_files = list(folder_dir.glob("energy_comparison/simulated_energy_vs_time*.csv"))
        if not csv_files:
            continue
        folder = csv_files[0]
        radius_folder = folder_dir.name
        
        folder_suffix = ""
        if "_g1e-06" in radius_folder:
            folder_suffix = "g1e-06"
        elif "_g1" in radius_folder:
            folder_suffix = "g1"

        target_suffix = ""
        if g_val == 1 or str(g_val) == "1" or g_val == "g1":
            target_suffix = "g1"
        elif g_val == "1e-06" or g_val == "g1e-06":
            target_suffix = "g1e-06"
            
        if folder_suffix != target_suffix:
            continue

        radius_cm = _radius_from_folder_name(radius_folder)
        diameter_mm = 2.0 * radius_cm * 10.0
        final_gold_energy_hJ, final_foam_energy_hJ = _load_final_gold_foam_energy(folder)
        rows.append((diameter_mm, final_gold_energy_hJ, final_foam_energy_hJ))

    if not rows:
        return np.array([]), np.array([]), np.array([])

    rows.sort(key=lambda item: item[0])
    diameters_mm = np.array([item[0] for item in rows], dtype=float)
    gold_energies_hJ = np.array([item[1] for item in rows], dtype=float)
    foam_energies_hJ = np.array([item[2] for item in rows], dtype=float)

    return diameters_mm, gold_energies_hJ, foam_energies_hJ


def plot_final_gold_energy_vs_diameter(*, data_root: Path = DATA_ROOT, figure_root: Path = FIGURE_ROOT) -> Path:
    """Create and save the diameter vs. final gold energy plot."""
    diameters_mm, gold_energies_hJ, foam_energies_hJ = gather_energy_sweep_results(data_root)
    diameters_mm_g1, gold_energies_hJ_g1, foam_energies_hJ_g1 = gather_energy_sweep_results(data_root, g_val=1)
    diameters_mm_g1e_06, gold_energies_hJ_g1e_06, foam_energies_hJ_g1e_06 = gather_energy_sweep_results(data_root, g_val="g1e-06")

    diameters_mm_ana_loss, gold_ana_loss, foam_ana_loss = gather_analytic_energy_sweep_results(ANALYTIC_ROOT, "E_wall_gold_loss_hJ", "E_foam_gold_loss_hJ")
    diameters_mm_ana_lam, gold_ana_lam, foam_ana_lam = gather_analytic_energy_sweep_results(ANALYTIC_ROOT, "E_wall_gold_loss_lam_eff_hJ", "E_foam_gold_loss_lam_eff_hJ")
    diameters_mm_ana_lam_p1, gold_ana_lam_p1, foam_ana_lam_p1 = gather_analytic_energy_sweep_results(ANALYTIC_ROOT, "E_wall_gold_loss_lam_eff_power_1_hJ", "E_foam_gold_loss_lam_eff_power_1_hJ")
    diameters_mm_ana_lam_p2, gold_ana_lam_p2, foam_ana_lam_p2 = gather_analytic_energy_sweep_results(ANALYTIC_ROOT, "E_wall_gold_loss_lam_eff_power_2_hJ", "E_foam_gold_loss_lam_eff_power_2_hJ")
    ensure_dir(figure_root)
    figure_path = figure_root / "final_gold_energy_vs_diameter.png"

    plt.rcParams["font.family"] = "serif"
    fig, main_ax = plt.subplots(figsize=(8, 5))
    
    # main_ax.plot(diameters_mm, gold_energies_hJ / foam_energies_hJ, marker="o", linewidth=2.0, color="tab:blue", label="x100 opaque Gold")
    if len(diameters_mm_g1) > 0:
        main_ax.plot(diameters_mm_g1, gold_energies_hJ_g1 / foam_energies_hJ_g1, marker="s", linewidth=2.0, color="tab:green", label="Nominal Gold (Sim)")
    # if len(diameters_mm_g1e_06) > 0:
    #     main_ax.plot(diameters_mm_g1e_06, gold_energies_hJ_g1e_06 / foam_energies_hJ_g1e_06, marker="^", linewidth=2.0, color="tab:red", label="g = 1e-06")
    
    if len(diameters_mm_ana_loss) > 0:
        main_ax.plot(diameters_mm_ana_loss, gold_ana_loss / foam_ana_loss, marker="x", linewidth=2.0, color="tab:purple", label="Analytic (Gold loss)")
    if len(diameters_mm_ana_lam) > 0:
        main_ax.plot(diameters_mm_ana_lam, gold_ana_lam / foam_ana_lam, marker="d", linewidth=2.0, color="tab:brown", label="Analytic (Gold loss + lam_eff)")
    if len(diameters_mm_ana_lam_p1) > 0:
        main_ax.plot(diameters_mm_ana_lam_p1, gold_ana_lam_p1 / foam_ana_lam_p1, marker="v", linewidth=2.0, color="tab:cyan", label="Analytic (Gold loss + lam_eff_1)")
    if len(diameters_mm_ana_lam_p2) > 0:
        main_ax.plot(diameters_mm_ana_lam_p2, gold_ana_lam_p2 / foam_ana_lam_p2, marker="^", linewidth=2.0, color="tab:pink", label="Analytic (Gold loss + lam_eff_2)")

    # print(gold_energies_hJ_g1 / (0.1 * np.pi * diameters_mm_g1))
    # print(foam_energies_hJ_g1)
    # print((0.1 * np.pi * diameters_mm_g1))
    main_ax.set_xlabel("Diameter (mm)")
    main_ax.set_ylabel("Final Gold Energy / Foam Energy")
    main_ax.grid(True, alpha=0.3)
    main_ax.legend()

    # Create inset axis
    ins_ax = main_ax.inset_axes([0.15, 0.4, 0.45, 0.45])
    # ins_ax.plot(diameters_mm, gold_energies_hJ  / foam_energies_hJ, marker="o", linewidth=2.0, color="tab:blue")
    if len(diameters_mm_g1) > 0:
        ins_ax.plot(diameters_mm_g1, gold_energies_hJ_g1 / foam_energies_hJ_g1, marker="s", linewidth=2.0, color="tab:green")
    # if len(diameters_mm_g1e_06) > 0:
    #     ins_ax.plot(diameters_mm_g1e_06, gold_energies_hJ_g1e_06 / foam_energies_hJ_g1e_06, marker="^", linewidth=2.0, color="tab:red")
    if len(diameters_mm_ana_loss) > 0:
        ins_ax.plot(diameters_mm_ana_loss, gold_ana_loss / foam_ana_loss, marker="x", linewidth=2.0, color="tab:purple")
    if len(diameters_mm_ana_lam) > 0:
        ins_ax.plot(diameters_mm_ana_lam, gold_ana_lam / foam_ana_lam, marker="d", linewidth=2.0, color="tab:brown")
    if len(diameters_mm_ana_lam_p1) > 0:
        ins_ax.plot(diameters_mm_ana_lam_p1, gold_ana_lam_p1 / foam_ana_lam_p1, marker="v", linewidth=2.0, color="tab:cyan")
    if len(diameters_mm_ana_lam_p2) > 0:
        ins_ax.plot(diameters_mm_ana_lam_p2, gold_ana_lam_p2 / foam_ana_lam_p2, marker="^", linewidth=2.0, color="tab:pink")

    ins_ax.set_xscale("log")
    ins_ax.set_xlim(5e-3, 2)
    ins_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {figure_path}")
    return figure_path
    

def plot_front_comparison_1p6mm(*, data_root: Path = DATA_ROOT, figure_root: Path = FIGURE_ROOT) -> Path:
    """Plot front position vs time for 1.6 mm diameter (radius 0.08 cm).
    
    Compares simulation (g=1 run) and analytic model (with and without effective lambda).
    """
    sim_dir = data_root / "R_0p08_g1"
    sim_csv = sim_dir / "front_vs_time" / "front_position_vs_time_Gold_r0.csv"
    
    ana_dir = ANALYTIC_ROOT / "R_0p08"
    ana_csv = ana_dir / "front_vs_time" / "analytic_positions.csv"
    
    if not sim_csv.exists():
        print(f"Warning: Simulation front CSV not found at {sim_csv}")
        return Path()
    if not ana_csv.exists():
        print(f"Warning: Analytic front CSV not found at {ana_csv}")
        return Path()
        
    sim_data = np.genfromtxt(sim_csv, delimiter=",", names=True)
    ana_data = np.genfromtxt(ana_csv, delimiter=",", names=True)
    
    ensure_dir(figure_root)
    figure_path = figure_root / "front_comparison_1p6mm.png"
    
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(9, 6))
    
    # Plot Simulation
    plt.plot(
        sim_data["time_ns"], 
        sim_data["front_position_mm"], 
        color="black", 
        linewidth=2.5, 
        linestyle="-", 
        label="Nominal Gold (Sim)",
        zorder=5
    )
    
    # Plot Analytic (without lam_eff)
    # Convert cm to mm by multiplying by 10.0
    plt.plot(
        ana_data["time_ns"], 
        ana_data["front_position_gold_loss_cm"] * 10.0, 
        color="tab:red", 
        linewidth=2.0, 
        linestyle="--", 
        label="Analytic (Gold loss, no lam_eff)"
    )
    
    # Plot Analytic (with lam_eff power 1)
    plt.plot(
        ana_data["time_ns"], 
        ana_data["front_position_gold_loss_lam_eff_power_1_cm"] * 10.0, 
        color="tab:green", 
        linewidth=2.0, 
        linestyle=":", 
        label="Analytic (Gold loss, lam_eff n=1)"
    )
    
    # Plot Analytic (with lam_eff power 2)
    plt.plot(
        ana_data["time_ns"], 
        ana_data["front_position_gold_loss_lam_eff_power_2_cm"] * 10.0, 
        color="tab:orange", 
        linewidth=2.0, 
        linestyle="-.", 
        label="Analytic (Gold loss, lam_eff n=2)"
    )
    
    # Plot Analytic (with lam_eff general)
    plt.plot(
        ana_data["time_ns"], 
        ana_data["front_position_gold_loss_lam_eff_cm"] * 10.0, 
        color="tab:purple", 
        linewidth=2.0, 
        linestyle="--", 
        dashes=(5, 2, 1, 2), # custom dash style to distinguish from standard dashed
        label="Analytic (Gold loss, lam_eff general)"
    )
    
    # Plot Marshak 1D if present (optional but nice)
    if "front_position_marshak_cm" in ana_data.dtype.names:
        plt.plot(
            ana_data["time_ns"], 
            ana_data["front_position_marshak_cm"] * 10.0, 
            color="gray", 
            linewidth=1.5, 
            linestyle="-", 
            alpha=0.6,
            label="Marshak 1D (No Loss)"
        )
        
    plt.xlabel("Time (ns)", fontsize=12)
    plt.ylabel("Front Position (mm)", fontsize=12)
    plt.title("Wave Front Position vs Time (1.6 mm Diameter / R = 0.08 cm)", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 3.0)
    plt.ylim(0, 1.8)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved front comparison figure: {figure_path}")
    return figure_path


def plot_energy_comparison_1p6mm(*, data_root: Path = DATA_ROOT, figure_root: Path = FIGURE_ROOT) -> Path:
    """Plot foam and gold energy vs time for 1.6 mm diameter (radius 0.08 cm).
    
    Compares simulation (g=1 run) and analytic model (with and without effective lambda).
    """
    sim_dir = data_root / "R_0p08_g1"
    sim_csv = sim_dir / "energy_comparison" / "simulated_energy_vs_time_gold.csv"
    
    ana_dir = ANALYTIC_ROOT / "R_0p08"
    ana_csv = ana_dir / "energy_comparison" / "analytic_energy_vs_time.csv"
    
    if not sim_csv.exists():
        print(f"Warning: Simulation energy CSV not found at {sim_csv}")
        return Path()
    if not ana_csv.exists():
        print(f"Warning: Analytic energy CSV not found at {ana_csv}")
        return Path()
        
    sim_data = np.genfromtxt(sim_csv, delimiter=",", names=True)
    ana_data = np.genfromtxt(ana_csv, delimiter=",", names=True)
    
    ensure_dir(figure_root)
    figure_path = figure_root / "energy_comparison_1p6mm.png"
    
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # --- FOAM ENERGIES (Blue/teal shades) ---
    # Simulation Foam
    ax.plot(
        sim_data["time_ns"], 
        sim_data["foam_energy_hJ"], 
        color="#0B5394", 
        linewidth=2.5, 
        linestyle="-", 
        label="Simulated Foam Energy",
        zorder=5
    )
    # Model Foam (No lam_eff)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_foam_gold_loss_hJ"], 
        color="#3D85C6", 
        linewidth=1.8, 
        linestyle="--", 
        label="Model Foam Energy (no lam_eff)"
    )
    # Model Foam (lam_eff n=1)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_foam_gold_loss_lam_eff_power_1_hJ"], 
        color="#6FA8DC", 
        linewidth=1.8, 
        linestyle=":", 
        label="Model Foam Energy (lam_eff n=1)"
    )
    # Model Foam (lam_eff n=2)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_foam_gold_loss_lam_eff_power_2_hJ"], 
        color="#45818E", 
        linewidth=1.8, 
        linestyle="-.", 
        label="Model Foam Energy (lam_eff n=2)"
    )
    # Model Foam (lam_eff general)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_foam_gold_loss_lam_eff_hJ"], 
        color="#134F5C", 
        linewidth=1.8, 
        linestyle="--", 
        dashes=(5, 2, 1, 2),
        label="Model Foam Energy (lam_eff general)"
    )
    
    # --- GOLD/WALL ENERGIES (Red/Orange/Brown shades) ---
    # Simulation Gold
    ax.plot(
        sim_data["time_ns"], 
        sim_data["gold_energy_hJ"], 
        color="#E69138", 
        linewidth=2.5, 
        linestyle="-", 
        label="Simulated Gold Energy",
        zorder=5
    )
    # Model Gold (No lam_eff)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_wall_gold_loss_hJ"], 
        color="#CD7F32", 
        linewidth=1.8, 
        linestyle="--", 
        label="Model Gold Energy (no lam_eff)"
    )
    # Model Gold (lam_eff n=1)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_wall_gold_loss_lam_eff_power_1_hJ"], 
        color="#CC0000", 
        linewidth=1.8, 
        linestyle=":", 
        label="Model Gold Energy (lam_eff n=1)"
    )
    # Model Gold (lam_eff n=2)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_wall_gold_loss_lam_eff_power_2_hJ"], 
        color="#E06666", 
        linewidth=1.8, 
        linestyle="-.", 
        label="Model Gold Energy (lam_eff n=2)"
    )
    # Model Gold (lam_eff general)
    ax.plot(
        ana_data["time_ns"], 
        ana_data["E_wall_gold_loss_lam_eff_hJ"], 
        color="#783F04", 
        linewidth=1.8, 
        linestyle="--", 
        dashes=(5, 2, 1, 2),
        label="Model Gold Energy (lam_eff general)"
    )
    
    # Plot E_marshak (1D Foam Energy without loss) for reference
    if "E_marshak_hJ" in ana_data.dtype.names:
        ax.plot(
            ana_data["time_ns"], 
            ana_data["E_marshak_hJ"], 
            color="gray", 
            linewidth=1.2, 
            linestyle="-", 
            alpha=0.5,
            label="Marshak 1D Energy"
        )
        
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Energy (hJ)", fontsize=12)
    ax.set_title("Foam and Gold Energy vs Time (1.6 mm Diameter / R = 0.08 cm)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 10.0)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved energy comparison figure: {figure_path}")
    return figure_path


def main() -> None:
    plot_front_time_vs_diameter()
    plot_final_gold_energy_vs_diameter()
    plot_front_comparison_1p6mm()
    plot_energy_comparison_1p6mm()


if __name__ == "__main__":
    main()
