
from pathlib import Path
import os
import sys

import matplotlib as mpl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csv_helpers import ensure_dir
from simulation_2d_core import SelfSimilarDiffusion2D, load_time_temp
from simulation_2d_plots import (
    plot_energy_comparison,
    plot_front_surface,
    plot_front_vs_time,
    plot_temperature_maps_gouraud,
    plot_temperature_maps_simple,
)

heatmap_times = (1e-9, 2e-9, 2.5e-9)  # default; overridden per-material below

Material = "SiO2_low_energy"
CoatingMaterial = "Gold"
if Material == "SiO2" or Material == "Ta2O5":
    Experiment = "Back"
    heatmap_times = (1e-9, 2e-9, 2.5e-9)
elif Material == "SiO2_low_energy":
    Experiment = "Back"
    heatmap_times = (5e-9, 8e-9, 11e-9)
elif Material == "C8H8":
    Experiment = "Ji-Yan"
    heatmap_times = (0.5e-9, 1e-9, 1.3e-9)
elif Material == "SiO2_copper" or Material == "SiO2_gold":
    Experiment = "French"
    heatmap_times = (1e-9, 2e-9, 2.5e-9)

BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "Data_new" / Experiment / Material
FIGURES_DIR = BASE_DIR / "Figures_new" / Experiment / Material
FIGURES_DIR_2D = FIGURES_DIR / "2D_simulation"
FIGURE_DATA_DIR_2D = DATA_DIR / "2D_simulation"

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.sans-serif"] = ["serif"]
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["mathtext.default"] = "regular"


def _default_dtmax_for_material(material: str) -> float:
    # Keep existing default for all materials, but allow larger max step for low-energy run.
    return 2e-11 if str(material) == "SiO2_low_energy" else 2e-12


def create_simulation(
    *,
    material: str = Material,
    coating_material: str = "Gold",
    R_foam: float | None = None,
    Nz: int = 320,
    Nr_foam: int = 320,
    kind_of_D_face: str = "arithmetic",
    chi: float = 1000.0,
    T_material_0_K: float = 300.0,
    gold_g_scale: float = 1,
):
    material = str(material)
    coating_material = str(coating_material)
    if material == "SiO2":
        # Foam self-similarity parameters
        f = 8.77 * 10**13
        g = 1 / 9175
        alpha = 3.53
        beta_exp = 1.1
        lambda_param = 0.75
        mu = 0.09
        rho = 0.05

        Lz = 0.3
        R_foam_default = 0.08

        csv_path = BASE_DIR / "Data_new" / Experiment / Material / "article" / "Temperatures" / "T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)

        t_final = 3e-9
        dt_init = 5e-15
 
    elif material == "C8H8":
        # Foam self-similarity parameters
        f = 21.17 * 10**13          # fudge factor for sigma (new model) [erg/g]
        g = 1 / 2818.1      
        alpha = 2.79       # opacity exponent
        beta_exp = 1.06    # beta exponent
        lambda_param = 0.81
        mu = 0.06
        rho = 0.16     # initial density (g/cm^3)
        R_foam_default = 0.01
        Lz = 0.03
        heatmap_times = (0.5e-9, 1e-9, 1.3e-9)
        csv_path = BASE_DIR / "Data_new" / Experiment / Material / "article" / "Temperatures" / "T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)
        print(csv_path)

        t_final = 1.31e-9
        dt_init = 5e-15
    
    elif material == "Ta2O5":
        f = 4.78 * 10**13          # fudge factor for sigma (new model) [erg/g]
        g = 1 / 8433.3      
        alpha = 1.78       # opacity exponent
        beta_exp = 1.37     # beta exponent
        lambda_param = 0.24
        mu = 0.12
        rho = 0.04      # initial density (g/cm^3)
        Lz = 0.3
        R_foam_default = 0.08

        csv_path = BASE_DIR / "Data_new" / Experiment / Material / "article" / "Temperatures" / "T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)
        print(csv_path)

        t_final = 3e-9
        dt_init = 5e-15

    elif material == "SiO2_gold":
        f = 8.77 * 10**13          # fudge factor for sigma (new model) [erg/g]
        g = 1 / 9175      
        alpha = 3.53     # opacity exponent
        beta_exp = 1.1       # beta exponent
        lambda_param = 0.75
        mu = 0.09
        rho = 0.029     # initial density (g/cm^3)
        R_foam_default = 0.05      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
        Lz = 0.5
        csv_path = BASE_DIR / "Data_new" / Experiment / Material / "article" / "Temperatures" / "T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)
        print(csv_path)

        t_final = 3e-9
        dt_init = 5e-15

    elif material == "SiO2_copper":
        f = 8.77 * 10**13          # fudge factor for sigma (new model) [erg/g]
        g = 1 / 9175      
        alpha = 3.53     # opacity exponent
        beta_exp = 1.1       # beta exponent
        lambda_param = 0.75
        mu = 0.09
        rho = 0.0189     # initial density (g/cm^3)
        R_foam_default = 0.1      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
        Lz = 0.4
        csv_path = BASE_DIR / "Data_new" / Experiment / Material / "article" / "Temperatures" / "T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)
        print(csv_path)

        t_final = 3e-9
        dt_init = 5e-15
    
    elif material == "SiO2_low_energy":
        f = 8.4 * 10**13          # fudge factor for sigma (new model) [erg/g]
        g = 1 / 9652      
        alpha = 2.0     # opacity exponent
        beta_exp = 1.23       # beta exponent
        lambda_param = 0.61
        mu = 0.1
        rho = 0.01     # initial density (g/cm^3)
        R_foam_default = 0.15      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
        Lz = 0.2
        csv_path = BASE_DIR / "Data_new" / Experiment / Material / "article" / "Temperatures" / "T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)
        t_final = 15e-9
        dt_init = 5e-14

    else:
        raise ValueError(f"{material} is not supported in this function for now.")

    if R_foam is None:
        R_foam = R_foam_default

    coating_key = coating_material.strip().lower()
    coating_width_map = {
        "gold": 25 * 1e-4,
        "copper": 3.5 * 1e-3,
        "be": 6e-3,
        "vacuum": 0.0,
    }
    coating_width = coating_width_map.get(coating_key)
    if coating_width is None:
        raise ValueError("coating_material must be one of 'Gold', 'Copper', 'Be', or 'Vacuum'.")

    foam_params = {
        "f": f,
        "g": g,
        "alpha": alpha,
        "beta_exp": beta_exp,
        "lambda_param": lambda_param,
        "mu": mu,
        "rho": rho,
    }
    gold_params = {
        "f": 3.4e13,
        "g": (1/7200) * float(gold_g_scale),
        "alpha": 1.5,
        "beta_exp": 1.6,
        "lambda_param": 0.2,
        "mu": 0.14,
        "rho": 19.32,
    }
    be_params = {
        "f": 8.81 * 10**13,
        "g": 1 / 402.8,
        "alpha": 4.893,
        "beta_exp": 1.0902,
        "lambda_param": 0.6726,
        "mu": 0.0701,
        "rho": 1.85,
    }
    copper_params = {
        "f": 5.7e13,
        "g": 4.47e-4,
        "alpha": 2.21,       # opacity exponent
        "beta_exp": 1.35,     # beta exponent
        "lambda_param": 0.29,
        "mu": 0.14,
        "rho": 8.96,      # initial density (g/cm^3)
    }

    coating_params_map = {
        "gold": gold_params,
        "copper": copper_params,
        "be": be_params,
        "vacuum": foam_params,
    }
    coating_params = coating_params_map[coating_key]


    return SelfSimilarDiffusion2D(
        Lz=Lz,
        gold_width=coating_width,
        R_foam=R_foam,
        Nz=int(Nz),
        Nr_foam=int(Nr_foam),
        dt_init=dt_init,
        t_final=t_final,
        simulation_unit_system="cgs",
        foam_params=foam_params,
        gold_params=gold_params,
        be_params=be_params,
        copper_params=copper_params,
        coating_params=coating_params,
        outer_material=coating_material,
        chi=float(chi),
        t_drive_ns=t_drive_ns,
        T_drive_eV=T_drive_eV,
        kind_of_D_face=str(kind_of_D_face),
        T_material_0_K=float(T_material_0_K),
    )


def _radius_tag(radius_cm: float) -> str:
    radius_cm = float(radius_cm)
    text = f"{radius_cm:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def run_eff_lam_radius_sweep(
    *,
    material: str = "SiO2",
    coating_material: str = "Gold",
    radii_cm: tuple[float, ...] | list[float] = (
        0.0004,
        0.0007,
        0.01,
        0.001,
        0.02,
        0.003,
        0.005,
        0.04,
        0.06,
        0.08,
        0.008,
        0.0025,
    ),
    Nz: int = 160,
    Nr_foam: int = 160,
    gold_g_scale: float = 1e-6,
    kind_of_D_face: str = "arithmetic",
    chi: float = 1000.0,
    T_material_0_K: float = 300.0,
    n_store: int = 50,
    store_start_frac: float = 0.01,
    dtfac: float = 0.05,
    dtmin: float | None = 5e-15,
    dtmax: float | None = None,
    bc_r_outer: str = "marshak_wall",
    marshak_boundary: bool = True,
):
    """Run a sweep over foam radii and save each run under an eff_lam folder."""

    figure_root = FIGURES_DIR_2D / "eff_lam"
    data_root = FIGURE_DATA_DIR_2D / "eff_lam"
    ensure_dir(figure_root)
    ensure_dir(data_root)

    if dtmax is None:
        dtmax = _default_dtmax_for_material(material)

    results = {}

    for radius_cm in radii_cm:
        radius_cm = float(radius_cm)
        radius_name = _radius_tag(radius_cm)
        print(f"Running eff_lam sweep for R_foam={radius_cm:.6g} cm")

        run_figures_dir = figure_root / f"R_{radius_name}_g{gold_g_scale}"
        run_data_dir = data_root / f"R_{radius_name}_g{gold_g_scale}"
        ensure_dir(run_figures_dir)
        ensure_dir(run_data_dir)

        sim = create_simulation(
            material=material,
            coating_material=coating_material,
            R_foam=radius_cm,
            Nz=Nz,
            Nr_foam=Nr_foam,
            kind_of_D_face=kind_of_D_face,
            chi=chi,
            T_material_0_K=T_material_0_K,
            gold_g_scale=gold_g_scale,
        )

        stored_t, stored_Um, stored_Tm, stored_TR = run_simulation(
            sim,
            n_store=n_store,
            store_start_frac=store_start_frac,
            dtfac=dtfac,
            dtmin=dtmin,
            dtmax=dtmax,
            bc_r_outer=bc_r_outer,
            marshak_boundary=marshak_boundary,
        )

        save_run_data(run_data_dir / "run_outputs.npz", stored_t, stored_Um, stored_Tm, stored_TR)

        front_z_cm = sim.compute_front_at_r(stored_Tm, r_index=0, front_method="maxgrad")

        plot_temperature_maps_gouraud(
            sim,
            stored_t,
            stored_Tm,
            times_s=heatmap_times,
            out_dir=run_figures_dir,
            figure_data_dir=run_data_dir,
        )
        print(heatmap_times)
        plot_temperature_maps_simple(
            sim,
            stored_t,
            stored_Tm,
            times_s=heatmap_times,
            out_dir=run_figures_dir,
            figure_data_dir=run_data_dir,
        )
        plot_front_vs_time(
            sim,
            stored_t,
            front_z_cm,
            out_path=run_figures_dir / "front_position - Front Position vs Time at r=0.png",
            figure_data_dir=run_data_dir,
            base_dir=BASE_DIR,
        )
        plot_front_surface(
            sim,
            stored_t,
            stored_Tm,
            times_s=heatmap_times,
            out_path=run_figures_dir / "front_surface - Front Surface zF vs r.png",
            figure_data_dir=run_data_dir,
        )
        plot_energy_comparison(
            sim,
            stored_t,
            stored_Um,
            out_path=run_figures_dir / "energy_comparison - Foam Energy vs Time.png",
            figure_data_dir=run_data_dir,
            base_dir=BASE_DIR,
            material=material,
            experiment=Experiment,
        )

        results[radius_cm] = {
            "sim": sim,
            "stored_t": stored_t,
            "stored_Um": stored_Um,
            "stored_Tm": stored_Tm,
            "stored_TR": stored_TR,
            "figure_dir": run_figures_dir,
            "data_dir": run_data_dir,
        }

    return results


def run_simulation(
    sim,
    *,
    n_store: int = 50,
    store_start_frac: float = 0.01,
    dtfac: float = 0.05,
    dtmin: float | None = 5e-15,
    dtmax: float | None = 2e-12,
    bc_r_outer: str = "marshak_wall",
    marshak_boundary: bool = True,
):
    """Run a simulation and return stored arrays (same as the old script)."""

    times_to_store = sim.t_final * np.linspace(float(store_start_frac), 1.0, int(n_store))
    stored_t, stored_Um, stored_Tm, stored_TR = sim.run(
        times_to_store,
        dtfac=float(dtfac),
        dtmin=dtmin,
        dtmax=dtmax,
        bc_r_outer = str(bc_r_outer), #"dirichlet_bath", "neumann0", or "marshak_wall",
        marshak_boundary=bool(marshak_boundary),
    )
    return stored_t, stored_Um, stored_Tm, stored_TR


def save_run_data(file_path, stored_t, stored_Um=None, stored_Tm=None, stored_TR=None):
    """Save (stored_t, stored_Um, stored_Tm, stored_TR) to a single .npz file.

    You can call either:
      - save_run_data(path, stored_t, stored_Um, stored_Tm, stored_TR)
      - save_run_data(path, (stored_t, stored_Um, stored_Tm, stored_TR))
    """

    if stored_Um is None and stored_Tm is None and stored_TR is None:
        stored_t, stored_Um, stored_Tm, stored_TR = stored_t

    file_path = str(file_path)
    if not file_path.lower().endswith(".npz"):
        file_path += ".npz"

    out_dir = os.path.dirname(file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        file_path,
        stored_t=np.asarray(stored_t),
        stored_Um=np.asarray(stored_Um),
        stored_Tm=np.asarray(stored_Tm),
        stored_TR=np.asarray(stored_TR),
    )
    return file_path


def run_default_pipeline(*, material: str = "SiO2", coating_material: str = "Gold", gold_g_scale: float = 1):
    sim = create_simulation(material=material, coating_material=coating_material, gold_g_scale=gold_g_scale)
    stored_t, stored_Um, stored_Tm, stored_TR = run_simulation(
        sim,
        n_store=50,
        store_start_frac=0.01,
        dtfac=0.05,
        dtmin=5e-15,
        dtmax=_default_dtmax_for_material(material),
        bc_r_outer="marshak_wall",
        marshak_boundary=True,
    )

    dir_suffix = f"_{gold_g_scale}" if gold_g_scale != 1.0 else ""
    current_data_dir_2d = DATA_DIR / f"2D{dir_suffix}"
    current_figures_dir = FIGURES_DIR / f"2D_simulation{dir_suffix}"
    current_figure_data_dir = DATA_DIR / f"2D_simulation{dir_suffix}"

    ensure_dir(current_data_dir_2d)
    save_run_data(current_data_dir_2d / "run_outputs.npz", stored_t, stored_Um, stored_Tm, stored_TR)

    front_z_cm = sim.compute_front_at_r(stored_Tm, r_index=0, front_method="maxgrad")

    plot_temperature_maps_gouraud(
        sim,
        stored_t,
        stored_Tm,
        times_s=heatmap_times,
        out_dir=current_figures_dir,
        figure_data_dir=current_figure_data_dir,
    )
    print(heatmap_times)
    plot_temperature_maps_simple(
        sim,
        stored_t,
        stored_Tm,
        times_s=heatmap_times,
        out_dir=current_figures_dir,
        figure_data_dir=current_figure_data_dir,
    )
    plot_front_vs_time(
        sim,
        stored_t,
        front_z_cm,
        out_path=current_figures_dir / "front_position - Front Position vs Time at r=0.png",
        figure_data_dir=current_figure_data_dir,
        base_dir=BASE_DIR,
    )
    plot_front_surface(
        sim,
        stored_t,
        stored_Tm,
        times_s=heatmap_times,
        out_path=current_figures_dir / "front_surface - Front Surface zF vs r.png",
        figure_data_dir=current_figure_data_dir,
    )
    plot_energy_comparison(
        sim,
        stored_t,
        stored_Um,
        out_path=current_figures_dir / "energy_comparison - Foam Energy vs Time.png",
        figure_data_dir=current_figure_data_dir,
        base_dir=BASE_DIR,
        material=material,
        experiment=Experiment,
    )

    # --- Flux Curvature and vs-Time Analysis ---
    from simulation_2d_flux import (
        plot_flux_curvature_post_breakout,
        compute_and_plot_flux_curvature,
        compute_and_plot_flux_vs_time,
    )

    flux_fig_dir = current_figures_dir / "flux"
    if material == "Ta2O5":
        detector_pos = [0.25, 0.5, 0.75, 1.0]
    elif material == "SiO2":
        detector_pos = [0.25, 0.5, 0.75, 1.0]
    else:
        detector_pos = [0.5, 1.0, 1.5]

    print("\n--- Running Flux vs Time Analysis ---")
    compute_and_plot_flux_vs_time(
        sim,
        stored_t,
        stored_Tm,
        detector_positions_mm=detector_pos,
        r_index=0,
        show_plot=False,
        out_dir=flux_fig_dir,
        title_suffix=f" — on-axis (r=0, {material})",
    )

    print("\n--- Running Flux Curvature Analysis (Snapshots) ---")
    snap_ns = [float(t) * 1e9 for t in heatmap_times]
    compute_and_plot_flux_curvature(
        sim,
        stored_t,
        stored_Tm,
        times_ns_snapshots=snap_ns,
        detector_positions_mm=detector_pos,
        show_plot=False,
        out_dir=flux_fig_dir,
        title_suffix=f" — {material}",
    )

    print("\n--- Running Post-Breakout Flux Curvature ---")
    plot_flux_curvature_post_breakout(
        sim,
        stored_t,
        stored_Tm,
        detector_positions_mm=detector_pos,
        delay_ns=0.5,
        show_plot=False,
        out_dir=flux_fig_dir,
        mode="Simulation",
        material=material,
    )

    heated_gold_cells_by_z = sim.compute_heated_gold_cells_by_z(stored_Tm)
    print("Heated outer-coat cells by z (last stored snapshot):")
    for z_cm, count in zip(sim.z, heated_gold_cells_by_z):
        if count > 0:
            print(f"  z={z_cm:.6g} cm: {int(count)} {sim.outer_material} cells")

    return sim, stored_t, stored_Um, stored_Tm, stored_TR
