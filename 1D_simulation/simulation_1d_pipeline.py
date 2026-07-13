from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csv_helpers import ensure_dir, save_series_csv
from parameters import Material, Experiment, K_per_Hev, a_hev, CGS
from simulation import GoldFoam1DSimulation


BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "Data_new" / Experiment / Material / "1D_simulation"
FIGURES_DIR = BASE_DIR / "Figures_new" / Experiment / Material / "1D_simulation"

SIGMA_SB_HEV = a_hev * 30.0 / 4.0
DETECTOR_POSITIONS_MM = [0.5, 1.0, 1.25]


def _gray_suffix_from_chi(chi_value):
    """Return filename suffix for gray (chi=1) runs."""
    try:
        return "_gray" if np.isclose(float(chi_value), 1.0) else ""
    except (TypeError, ValueError):
        return ""


def _gray_suffix_from_sim(sim):
    return _gray_suffix_from_chi(getattr(sim, "chi", None))


def create_simulation(
    *,
    nz: int = 800,
    lz: float | None = None,
    t_final: float | None = None,
    kind_of_D_face: str | None = None,
    chi: float | None = 1000,
    gold_block_width: float | None = 0,
):
    """Create the default 1D simulation object."""
    return GoldFoam1DSimulation(
        nz=nz,
        lz=lz,
        t_final_override=t_final,
        kind_of_D_face_override=kind_of_D_face,
        chi_override=chi,
        gold_block_width=gold_block_width,
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
    right_boundary: str = "dirichlet_cold",
):
    """Run a 1D simulation and return stored arrays."""
    times_to_store = sim.t_final * np.linspace(float(store_start_frac), 1.0, int(n_store))
    stored_t, stored_Um, stored_Tm, stored_TR = sim.run(
        times_to_store,
        dtfac=float(dtfac),
        dtmin=dtmin,
        dtmax=dtmax,
        marshak_boundary=bool(marshak_boundary),
        right_boundary=right_boundary,
    )
    info = sim.get_info()
    print(info)
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


def plot_run_outputs(sim, stored_t, stored_Tm, front_positions, foam_energies, gold_energies, material=None, filename_suffix=""):
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
    # ax.set_xlim(0, 0.005)
    ax.grid(True)
    if n_curves > 0:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"temperature_profiles{filename_suffix}.png", dpi=200)
    plt.close(fig)

    #print the tempertures of all cells at finish time
    print(f"Final time: {stored_t[-1]:.6g} ns")
    print(f"Final temperatures (HeV): {stored_Tm[-1]}")

    # 2) Front position vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stored_t, front_positions)
    # add here the article data from Data_new\experiment\material\article\fronts
    csv_path = BASE_DIR / "Data_new" / Experiment / material / "article" / "fronts" / "HR_simple.csv"
    if csv_path.exists():
        try:
            data = np.genfromtxt(csv_path, delimiter=',', names=True)
            ax.plot(data['x'], data['y']/10, '--', label="Article - HR Simple")
            ax.legend()
        except Exception as e:
            print(f"Could not load article data: {e}")
    csv_path = BASE_DIR / "Data_new" / Experiment / material / "article" / "fronts" / "HR_eff_1D.csv"
    if csv_path.exists():
        try:
            data = np.genfromtxt(csv_path, delimiter=',', names=True)
            ax.plot(data['x'], data['y']/10, '--', label="Article - HR Effective 1D")
            ax.legend()
        except Exception as e:
            print(f"Could not load article data: {e}")
    csv_path = BASE_DIR / "Data_new" / Experiment / material / "article" / "fronts" / "gold_wall.csv"
    if csv_path.exists():
        try:
            data = np.genfromtxt(csv_path, delimiter=',', names=True)
            ax.plot(data['x'], data['y']/10, '--', label="Article - Gold Wall")
            ax.legend()
        except Exception as e:
            print(f"Could not load article data: {e}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Front Position (cm)")
    ax.set_title("Front Position vs Time")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"front_position{filename_suffix}.png", dpi=200)
    plt.close(fig)

    # 3) Total material energy vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if material == "SiO2":
        ax.plot(stored_t, foam_energies, label="Foam")
        ax.plot(stored_t, gold_energies, label="Gold")
        csv_path = BASE_DIR / "Data_new" / Experiment / material / "article" / "energies" / "total_energy_1D.csv"
        if csv_path.exists():
            try:
                data = np.genfromtxt(csv_path, delimiter=',', names=True)
                ax.plot(data['x'], data['y'], '--', label="Article - 1D")
                ax.legend()
            except Exception as e:
                print(f"Could not load article data: {e}")
        csv_path = BASE_DIR / "Data_new" / Experiment / material / "article" / "energies" / "total_energy_2D.csv"
        if csv_path.exists():
            try:
                data = np.genfromtxt(csv_path, delimiter=',', names=True)
                ax.plot(data['x'], data['y'], '--', label="Article - 2D")
                ax.legend()
            except Exception as e:
                print(f"Could not load article data: {e}")
        csv_path = BASE_DIR / "Data_new" / Experiment / material / "article" / "energies" / "gold_wall_flattop.csv"
        if csv_path.exists():
            try:
                data = np.genfromtxt(csv_path, delimiter=',', names=True)
                ax.plot(data['x'], data['y'], '--', label="Article - Gold Wall Flattop")
                ax.legend()
            except Exception as e:
                print(f"Could not load article data: {e}")
    else:
        ax.plot(stored_t, foam_energies, label="Foam")
        ax.plot(stored_t, gold_energies, label="Gold")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Energy (hJ/mm$^2$)")
    ax.set_title("Total Material Energy vs Time")
    #y_lim
    #ax.set_ylim(0, 0.05)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"total_energy{filename_suffix}.png", dpi=200)
    plt.close(fig)


def compute_flux_vs_time(sim, stored_t, stored_Tm, *, z_pos_cm, threshold_factor=10):
    """Compute the radiation flux history at a single detector position."""
    stored_t = np.asarray(stored_t, dtype=float)
    stored_Tm = np.asarray(stored_Tm, dtype=float)
    z_idx = int(np.argmin(np.abs(sim.z - z_pos_cm)))
    T_hev = stored_Tm[:, z_idx]

    flux_raw = SIGMA_SB_HEV * T_hev ** 4
    T_cold = 300.0 / K_per_Hev
    mask = T_hev > (threshold_factor * T_cold)
    if np.any(mask):
        i_arrival = int(np.argmax(mask))
        t_arrival = stored_t[i_arrival]
    else:
        i_arrival = None
        t_arrival = None

    return {
        "times": stored_t,
        "times_ns": stored_t,
        "z_pos_cm": z_pos_cm,
        "z_pos_mm": z_pos_cm * 10.0,
        "T_hev": T_hev,
        "flux_raw": flux_raw,
        "i_arrival": i_arrival,
        "t_arrival": t_arrival,
    }


def compute_detector_end_flux_vs_time(
    reference_sim,
    times_to_store,
    *,
    z_pos_cm,
    threshold_factor=10,
    dtfac=0.05,
    dtmin=5e-15,
    dtmax=2e-13,
):
    """Run a detector-length tube and compute the outgoing vacuum Marshak flux."""
    z_pos_cm = float(z_pos_cm)
    if z_pos_cm <= 0.0:
        raise ValueError("Detector position must be positive.")

    dz_ref = getattr(reference_sim, "dz", z_pos_cm)
    nz = max(2, int(round(z_pos_cm / dz_ref)) + 1)
    flux_sim = create_simulation(
        nz=nz,
        lz=z_pos_cm,
        t_final=float(reference_sim.t_final),
        kind_of_D_face=reference_sim.kind_of_D_face,
        chi=reference_sim.chi,
        gold_block_width=0,
    )
    times_to_store = np.asarray(times_to_store, dtype=float)
    times_to_store_internal = times_to_store / 1e9 if flux_sim.simulation_unit_system == CGS else times_to_store
    stored_t, _stored_Um, _stored_Tm, stored_TR = flux_sim.run(
        times_to_store_internal,
        dtfac=float(dtfac),
        dtmin=dtmin,
        dtmax=dtmax,
        marshak_boundary=True,
        right_boundary="marshak_vacuum",
    )

    stored_TR = np.asarray(stored_TR, dtype=float)
    if stored_TR.ndim != 2 or stored_TR.shape[0] == 0:
        raise RuntimeError(
            "Detector-length flux simulation did not store any radiation-temperature profiles. "
            "Check the requested storage times and unit system."
        )

    T_hev = np.asarray(stored_TR[:, -1], dtype=float)
    flux_raw = 2.0 * SIGMA_SB_HEV * T_hev ** 4
    T_cold = 300.0 / K_per_Hev
    mask = T_hev > (threshold_factor * T_cold)
    if np.any(mask):
        i_arrival = int(np.argmax(mask))
        t_arrival = stored_t[i_arrival]
    else:
        i_arrival = None
        t_arrival = None

    return {
        "times": stored_t,
        "times_ns": stored_t,
        "z_pos_cm": z_pos_cm,
        "z_pos_mm": z_pos_cm * 10.0,
        "T_hev": T_hev,
        "flux_raw": flux_raw,
        "i_arrival": i_arrival,
        "t_arrival": t_arrival,
        "flux_sim": flux_sim,
    }


def compute_flux_datasets(sim, stored_t, stored_Tm, *, detector_positions_mm=None):
    """Compute and normalise detector-end flux curves from separate tube runs."""
    if detector_positions_mm is None:
        detector_positions_mm = DETECTOR_POSITIONS_MM

    datasets = []
    for z_mm in detector_positions_mm:
        ds = compute_detector_end_flux_vs_time(sim, stored_t, z_pos_cm=z_mm / 10.0)
        datasets.append(ds)

    global_peak = max((float(np.max(ds["flux_raw"])) for ds in datasets), default=0.0)
    for ds in datasets:
        if global_peak <= 0.0:
            ds["flux_normalised"] = np.zeros_like(ds["flux_raw"])
        else:
            ds["flux_normalised"] = ds["flux_raw"] / global_peak

    return datasets


def save_flux_datasets_csv(datasets, *, out_path=None, filename_suffix=""):
    """Save flux datasets in long-form CSV format."""
    if not datasets:
        return None

    if out_path is None:
        out_path = DATA_DIR / f"flux_vs_time{filename_suffix}.csv"

    time_ns = []
    detector_mm = []
    flux_raw = []
    flux_normalised = []
    temperature_hev = []
    arrival_time_ns = []

    for ds in datasets:
        times = np.asarray(ds["times_ns"], dtype=float)
        detector_value = float(ds["z_pos_mm"])
        peak_time = np.nan if ds["t_arrival"] is None else float(ds["t_arrival"])

        time_ns.extend(times.tolist())
        detector_mm.extend([detector_value] * len(times))
        flux_raw.extend(np.asarray(ds["flux_raw"], dtype=float).tolist())
        flux_normalised.extend(np.asarray(ds["flux_normalised"], dtype=float).tolist())
        temperature_hev.extend(np.asarray(ds["T_hev"], dtype=float).tolist())
        arrival_time_ns.extend([peak_time] * len(times))

    save_series_csv(
        out_path,
        {
            "time_ns": time_ns,
            "detector_mm": detector_mm,
            "flux_raw": flux_raw,
            "flux_normalised": flux_normalised,
            "temperature_hev": temperature_hev,
            "arrival_time_ns": arrival_time_ns,
        },
    )
    return Path(out_path)


def plot_flux_vs_time(datasets, *, out_dir=None, title_suffix="", filename_suffix=""):
    """Plot raw and normalised flux alongside detector temperatures."""
    if not datasets:
        return

    if out_dir is None:
        out_dir = FIGURES_DIR / "flux"
    ensure_dir(out_dir)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(datasets)))

    for ds, color in zip(datasets, colors):
        label = f"z={ds['z_pos_mm']:.1f} mm"
        axes[0].plot(ds["times_ns"], ds["flux_raw"], color=color, linewidth=1.6, label=label)
        axes[1].plot(ds["times_ns"], ds["flux_normalised"], color=color, linewidth=1.6, label=label)
        axes[2].plot(ds["times_ns"], ds["T_hev"], color=color, linewidth=1.6, label=label)

    axes[0].set_ylabel("Raw flux")
    axes[1].set_ylabel("Normalised flux")
    axes[2].set_ylabel("T (HeV)")
    axes[2].set_xlabel("Time (ns)")

    axes[0].set_title("1D Radiation Flux")
    axes[1].set_title("Normalised Radiation Flux")
    axes[2].set_title("Detector Temperature")

    for ax in axes:
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle(f"1D Simulation Flux Analysis{title_suffix}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"flux_vs_time{filename_suffix}.png", dpi=200)
    plt.close(fig)


def compute_and_plot_flux_vs_time(sim, stored_t, stored_Tm, *, detector_positions_mm=None, out_dir=None, title_suffix="", filename_suffix=""):
    """Convenience wrapper that computes, prints, and plots 1D flux curves."""
    datasets = compute_flux_datasets(sim, stored_t, stored_Tm, detector_positions_mm=detector_positions_mm)

    print("\n" + "=" * 72)
    print(f"{'z [mm]':>8}  {'t_arrival':>14}  {'Peak Φ':>16}  {'Peak T [HeV]':>14}")
    print("-" * 72)
    for ds in datasets:
        t_arr = ds["t_arrival"]
        t_str = f"{t_arr:.4g}" if t_arr is not None else "N/A"
        peak_flux = float(np.max(ds["flux_raw"]))
        peak_T = float(np.max(ds["T_hev"]))
        print(f"{ds['z_pos_mm']:8.1f}  {t_str:>14}  {peak_flux:16.4e}  {peak_T:14.4f}")
    print("=" * 72 + "\n")

    plot_flux_vs_time(datasets, out_dir=out_dir, title_suffix=title_suffix, filename_suffix=filename_suffix)
    return datasets


def run_default_pipeline(*, material: str = Material):
    """Default 1D pipeline, shaped like the 2D run_default_pipeline API."""
    sim = create_simulation()
    sim.data_dir = str(DATA_DIR)
    file_suffix = _gray_suffix_from_sim(sim)
    stored_t, stored_Um, stored_Tm, stored_TR = run_simulation(sim)

    ensure_dir(DATA_DIR)
    save_run_data(DATA_DIR / f"run_outputs_1d{file_suffix}.npz", stored_t, stored_Um, stored_Tm, stored_TR)
    sim.save_outputs(stored_t, stored_Um, stored_Tm, stored_TR, marshak_boundary=True)

    front_positions, foam_energies, gold_energies = sim.compute_front_and_energy(stored_Um, stored_Tm)
    plot_run_outputs(
        sim,
        stored_t,
        stored_Tm,
        front_positions,
        foam_energies,
        gold_energies,
        material=material,
        filename_suffix=file_suffix,
    )
    # flux_fig_dir = FIGURES_DIR / "flux"
    # flux_datasets = compute_and_plot_flux_vs_time(
    #     sim,
    #     stored_t,
    #     stored_Tm,
    #     detector_positions_mm=DETECTOR_POSITIONS_MM,
    #     out_dir=flux_fig_dir,
    #     title_suffix=f" — {material}",
    # )
    # flux_csv_path = save_flux_datasets_csv(flux_datasets)

    return {
        "sim": sim,
        "stored_t": stored_t,
        "stored_Um": stored_Um,
        "stored_Tm": stored_Tm,
        "stored_TR": stored_TR,
        "front_positions": front_positions,
        "foam_energies": foam_energies,
        "gold_energies": gold_energies,
        # "flux_datasets": flux_datasets,
        # "flux_csv_path": flux_csv_path,
        "material": material,
        "data_dir": DATA_DIR,
        "figures_dir": FIGURES_DIR,
    }
