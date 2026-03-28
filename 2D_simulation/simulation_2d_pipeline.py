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

Material = "SiO2"
Experiment = "Back"

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


def create_simulation(
    *,
    material: str = "SiO2",
    Nz: int = 10,
    Nr_foam: int = 10,
    kind_of_D_face: str = "arithmetic",
    chi: float = 1000.0,
    T_material_0_K: float = 300.0,
):
    material = str(material)
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
        R_foam = 0.08
        gold_width = 25 * 1e-4

        csv_path = BASE_DIR / "Data_new" / Experiment / Material / "article" / "Temperatures" / "T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)

        t_final = 3e-9
        dt_init = 5e-15

    else:
        raise ValueError(f"{material} is not supported in this function for now.")

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
        "g": 1/7200,
        "alpha": 1.5,
        "beta_exp": 1.6,
        "lambda_param": 0.2,
        "mu": 0.14,
        "rho": 19.32,
    }

    return SelfSimilarDiffusion2D(
        Lz=Lz,
        gold_width=gold_width,
        R_foam=R_foam,
        Nz=int(Nz),
        Nr_foam=int(Nr_foam),
        dt_init=dt_init,
        t_final=t_final,
        simulation_unit_system="cgs",
        foam_params=foam_params,
        gold_params=gold_params,
        chi=float(chi),
        t_drive_ns=t_drive_ns,
        T_drive_eV=T_drive_eV,
        kind_of_D_face=str(kind_of_D_face),
        T_material_0_K=float(T_material_0_K),
    )


def run_simulation(
    sim,
    *,
    n_store: int = 50,
    store_start_frac: float = 0.01,
    dtfac: float = 0.05,
    dtmin: float | None = 5e-15,
    dtmax: float | None = 2e-12,
    bc_r_outer: str = "dirichlet_bath",
    marshak_boundary: bool = True,
):
    """Run a simulation and return stored arrays (same as the old script)."""

    times_to_store = sim.t_final * np.linspace(float(store_start_frac), 1.0, int(n_store))
    stored_t, stored_Um, stored_Tm, stored_TR = sim.run(
        times_to_store,
        dtfac=float(dtfac),
        dtmin=dtmin,
        dtmax=dtmax,
        bc_r_outer=str(bc_r_outer),
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


def run_default_pipeline(*, material: str = "SiO2"):
    sim = create_simulation(material=material)
    stored_t, stored_Um, stored_Tm, stored_TR = run_simulation(
        sim,
        n_store=50,
        store_start_frac=0.01,
        dtfac=0.05,
        dtmin=5e-15,
        dtmax=2e-12,
        bc_r_outer="dirichlet_bath",
        marshak_boundary=True,
    )

    ensure_dir(DATA_DIR / "2D")
    save_run_data(DATA_DIR / "2D" / "run_outputs.npz", stored_t, stored_Um, stored_Tm, stored_TR)

    front_z_cm = sim.compute_front_at_r(stored_Tm, r_index=0, front_method="maxgrad")

    plot_temperature_maps_gouraud(
        sim,
        stored_t,
        stored_Tm,
        out_dir=FIGURES_DIR_2D,
        figure_data_dir=FIGURE_DATA_DIR_2D,
    )
    plot_temperature_maps_simple(
        sim,
        stored_t,
        stored_Tm,
        out_dir=FIGURES_DIR_2D,
        figure_data_dir=FIGURE_DATA_DIR_2D,
    )
    plot_front_vs_time(
        stored_t,
        front_z_cm,
        out_path=FIGURES_DIR_2D / "front_position - Front Position vs Time at r=0.png",
        figure_data_dir=FIGURE_DATA_DIR_2D,
        base_dir=BASE_DIR,
    )
    plot_front_surface(
        sim,
        stored_t,
        stored_Tm,
        out_path=FIGURES_DIR_2D / "front_surface - Front Surface zF vs r.png",
        figure_data_dir=FIGURE_DATA_DIR_2D,
    )
    plot_energy_comparison(
        sim,
        stored_t,
        stored_Um,
        out_path=FIGURES_DIR_2D / "energy_comparison - Foam Energy vs Time.png",
        figure_data_dir=FIGURE_DATA_DIR_2D,
        base_dir=BASE_DIR,
        material=Material,
        experiment=Experiment,
    )

    return sim, stored_t, stored_Um, stored_Tm, stored_TR
