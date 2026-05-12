from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csv_helpers import ensure_dir
from parameters import Material
from simulation import GoldFoam1DSimulation


BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "Data_new" / "Back" / "SiO2" / "1D_simulation"
FIGURES_DIR = BASE_DIR / "Figures_new" / "Back" / "SiO2" / "1D_simulation"


def create_simulation(
    *,
    t_final: float | None = None,
    kind_of_D_face: str | None = None,
    chi: float | None = None,
):
    """Create the default 1D simulation object."""
    return GoldFoam1DSimulation(
        t_final_override=t_final,
        kind_of_D_face_override=kind_of_D_face,
        chi_override=chi,
    )


def run_simulation(
    sim,
    *,
    n_store: int = 150,
    store_start_frac: float = 0.01,
    dtfac: float = 0.05,
    dtmin: float | None = 5e-15,
    dtmax: float | None = 2e-13,
    marshak_boundary: bool = True,
):
    """Run a 1D simulation and return stored arrays."""
    times_to_store = sim.t_final * np.linspace(float(store_start_frac), 1.0, int(n_store))
    stored_t, stored_Um, stored_Tm, stored_TR = sim.run(
        times_to_store,
        dtfac=float(dtfac),
        dtmin=dtmin,
        dtmax=dtmax,
        marshak_boundary=bool(marshak_boundary),
    )
    return stored_t, stored_Um, stored_Tm, stored_TR


def save_run_data(file_path, stored_t, stored_Um=None, stored_Tm=None, stored_TR=None):
    """Save (stored_t, stored_Um, stored_Tm, stored_TR) to a compressed .npz file."""
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


def plot_run_outputs(sim, stored_t, stored_Tm, front_positions, total_energies, material=None):
    """Save a compact set of standard 1D run plots."""
    ensure_dir(FIGURES_DIR)

    # 1) Temperature profiles vs z for stored snapshots
    fig, ax = plt.subplots(figsize=(8, 5))
    n_curves = min(8, len(stored_t))
    if n_curves > 0:
        indices = np.linspace(0, len(stored_t) - 1, n_curves).astype(int)
        for idx in indices:
            ax.plot(sim.z, stored_Tm[idx], label=f"t={stored_t[idx]:.3g} ns")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("T (HeV)")
    ax.set_title("1D Temperature Profiles")
    ax.grid(True)
    if n_curves > 0:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temperature_profiles.png", dpi=200)
    plt.close(fig)

    # 2) Front position vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stored_t, front_positions)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Front Position (cm)")
    ax.set_title("Front Position vs Time")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "front_position.png", dpi=200)
    plt.close(fig)

    # 3) Total material energy vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if material == "SiO2":
        ax.plot(stored_t, total_energies, label="Simulation")
        csv_path = BASE_DIR / "Data_new" / "Back" / "SiO2" / "article" / "energies" / "total_energy_1D.csv"
        if csv_path.exists():
            try:
                data = np.genfromtxt(csv_path, delimiter=',', names=True)
                ax.plot(data['x'], data['y'], '--', label="Article")
                ax.legend()
            except Exception as e:
                print(f"Could not load article data: {e}")
    else:
        ax.plot(stored_t, total_energies)
        
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Energy (hJ/mm^2)")
    ax.set_title("Total Material Energy vs Time")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "total_energy.png", dpi=200)
    plt.close(fig)


def run_default_pipeline(*, material: str = Material):
    """Default 1D pipeline, shaped like the 2D run_default_pipeline API."""
    sim = create_simulation()
    sim.data_dir = str(DATA_DIR)
    stored_t, stored_Um, stored_Tm, stored_TR = run_simulation(sim)

    ensure_dir(DATA_DIR)
    save_run_data(DATA_DIR / "run_outputs_1d.npz", stored_t, stored_Um, stored_Tm, stored_TR)
    sim.save_outputs(stored_t, stored_Um, stored_Tm, stored_TR, marshak_boundary=True)

    front_positions, total_energies = sim.compute_front_and_energy(stored_Um, stored_Tm)
    plot_run_outputs(sim, stored_t, stored_Tm, front_positions, total_energies, material=material)

    return {
        "sim": sim,
        "stored_t": stored_t,
        "stored_Um": stored_Um,
        "stored_Tm": stored_Tm,
        "stored_TR": stored_TR,
        "front_positions": front_positions,
        "total_energies": total_energies,
        "material": material,
        "data_dir": DATA_DIR,
        "figures_dir": FIGURES_DIR,
    }
