"""
Radiation Flux Computation & Normalization at Fixed Spatial Locations
=====================================================================

Computes the radiation flux Φ(t) = σ_SB · T(z,t)⁴  at three fixed
detector positions (z = 0.5, 1.0, 1.5 mm) using the self-similar
Henyey-like temperature profile from the Marshak wave-front solver.

Temperature profile (behind the front):
    T(z, t) = Ts(t) · (1 − z / z_F(t))^{1/(4+α−β)}

Ahead of the front (z ≥ z_F(t)), T = 0.

The flux is tracked from t = 0 for each location, but the arrival
time t_start (when z_F first reaches z) is also detected and stored.

All three curves are then normalised to [0, 1] for relative comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import alpha, beta, K_per_Hev, Experiment, Material, a_hev, c
from model_main import analytic_wave_front_dispatch, BASE_DIR
from csv_helpers import save_figure

# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------
# SIGMA_SB = 5.670374e-8  # Stefan-Boltzmann constant [W / m² K⁴]
SIGMA_SB = a_hev * c / 4.0  # Stefan-Boltzmann constant in HeV [W / m² K⁴]



# --------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------

def henyey_temperature(z_pos_cm, z_front_cm, Ts_hev, alpha_val, beta_val):
    """
    Compute the Henyey-like self-similar temperature at a single
    spatial position *z_pos_cm* given the current front position
    *z_front_cm* and surface temperature *Ts_hev* (in HeV).

    Returns T in HeV.  Returns 0 when z_pos >= z_front (ahead of the front).
    """
    if z_front_cm <= 0.0 or z_pos_cm >= z_front_cm:
        return 0.0

    exponent = 1.0 / (4.0 + alpha_val - beta_val)
    ratio = 1.0 - z_pos_cm / z_front_cm
    # Guard against tiny negative ratios from floating-point noise
    ratio = max(ratio, 0.0)
    return Ts_hev * (ratio ** exponent)


def henyey_temperature_array(z_pos_cm, z_front_array, Ts_array, alpha_val, beta_val):
    """
    Vectorised version of *henyey_temperature* over full time arrays.

    Parameters
    ----------
    z_pos_cm : float
        Fixed spatial location [cm].
    z_front_array : 1-D array
        Heat-front position z_F(t) for every time step [cm].
    Ts_array : 1-D array
        Surface temperature Ts(t) for every time step [HeV].
    alpha_val, beta_val : float
        Foam opacity / EOS exponents.

    Returns
    -------
    T_hev : 1-D array
        Temperature at *z_pos_cm* for every time step [HeV].
    """
    exponent = 1.0 / (4.0 + alpha_val - beta_val)
    ratio = 1.0 - z_pos_cm / np.where(z_front_array > 0, z_front_array, 1.0)
    ratio = np.clip(ratio, 0.0, None)

    # Mask: the front has not yet reached z_pos
    behind_front = z_front_array > z_pos_cm

    T_hev = np.where(behind_front, Ts_array * (ratio ** exponent), 0.0)
    return T_hev


# --------------------------------------------------------------------------
# Arrival-time detection
# --------------------------------------------------------------------------

def detect_arrival_time(z_pos_cm, times, z_front_array):
    """
    Detect the first time index at which the heat-front reaches *z_pos_cm*.

    Returns
    -------
    i_start : int or None
        Index into *times* / *z_front_array* where z_F first >= z_pos.
    t_start : float or None
        Corresponding time value (same units as *times*).
    """
    mask = z_front_array >= z_pos_cm
    if not np.any(mask):
        return None, None
    i_start = int(np.argmax(mask))
    return i_start, times[i_start]


# --------------------------------------------------------------------------
# Flux computation
# --------------------------------------------------------------------------

def compute_flux_at_position(z_pos_cm, times, z_front_array, Ts_array,
                             alpha_val=None, beta_val=None):
    """
    Compute the radiation flux Φ(t) = σ_SB · T(z,t)⁴  at a fixed spatial
    location, for all time steps.

    * Temperature is obtained from the Henyey profile.
    * Before the front arrives T = 0  ⇒  Φ = 0.

    Parameters
    ----------
    z_pos_cm : float
        Detector location [cm].
    times : 1-D array
        Time array [ns] (or seconds — just needs to be consistent).
    z_front_array : 1-D array
        Heat-front position z_F(t) [cm].
    Ts_array : 1-D array
        Surface temperature Ts(t) [HeV].
    alpha_val, beta_val : float, optional
        Foam exponents.  Default to the global *alpha*, *beta*.

    Returns
    -------
    dict with keys:
        'times'         – input time array
        'z_pos_cm'      – detector position [cm]
        'z_pos_mm'      – detector position [mm]
        'T_hev'         – temperature at the detector [HeV] (full array)
        'flux_raw'      – un-normalised flux Φ(t) [W/m²]
        'i_arrival'     – index of first arrival
        't_arrival'     – arrival time (same units as *times*)
    """
    if alpha_val is None:
        alpha_val = alpha
    if beta_val is None:
        beta_val = beta

    # 1. Temperature profile at the detector position
    T_hev = henyey_temperature_array(z_pos_cm, z_front_array, Ts_array,
                                     alpha_val, beta_val)

    # 2. Stefan-Boltzmann flux
    flux = SIGMA_SB * T_hev ** 4  # [W / m²]

    # 3. Arrival detection
    i_arr, t_arr = detect_arrival_time(z_pos_cm, times, z_front_array)

    return {
        'times':      times,
        'z_pos_cm':   z_pos_cm,
        'z_pos_mm':   z_pos_cm * 10.0,
        'T_hev':      T_hev,
        'flux_raw':   flux,
        'i_arrival':  i_arr,
        't_arrival':  t_arr,
    }

# --------------------------------------------------------------------------
# High-level driver: three detectors
# --------------------------------------------------------------------------

DETECTOR_POSITIONS_MM = [0.5, 1.0, 1.5]  # mm


def compute_radiation_flux_datasets(
    times,
    z_front_array,
    Ts_array,
    *,
    detector_positions_mm=None,
    alpha_val=None,
    beta_val=None,
):
    """
    Compute, normalise, and package radiation-flux data for multiple
    detector positions.

    Parameters
    ----------
    times : 1-D array
        Time array [ns].
    z_front_array : 1-D array
        Heat-front position z_F(t) [cm].
    Ts_array : 1-D array
        Surface temperature Ts(t) [HeV].
    detector_positions_mm : list of float, optional
        Detector locations in **mm**.  Default: [0.5, 1.0, 1.5].
    alpha_val, beta_val : float, optional
        Foam exponents.

    Returns
    -------
    datasets : list of dict
        One dict per detector position.  Each dict contains all output
        from *compute_flux_at_position* plus:
            'flux_normalised' – flux normalised to [0, 1]
    """
    if detector_positions_mm is None:
        detector_positions_mm = DETECTOR_POSITIONS_MM

    datasets = []
    for z_mm in detector_positions_mm:
        z_cm = z_mm / 10.0  # mm → cm
        result = compute_flux_at_position(
            z_cm, times, z_front_array, Ts_array,
            alpha_val=alpha_val, beta_val=beta_val,
        )
        datasets.append(result)

    # Find the global maximum peak over all detector positions
    global_peak = 0.0
    if len(datasets) > 0:
        global_peak = max(np.max(res['flux_raw']) for res in datasets)

    # Normalize all curves using the same global peak factor (preserving relative magnitude)
    for res in datasets:
        if global_peak <= 0.0:
            res['flux_normalised'] = np.zeros_like(res['flux_raw'])
        else:
            res['flux_normalised'] = res['flux_raw'] / global_peak

    return datasets


# --------------------------------------------------------------------------
# Quick summary printer
# --------------------------------------------------------------------------

def print_flux_summary(datasets):
    """Print a table of arrival times and peak fluxes for each detector."""
    print("\n" + "=" * 72)
    print(f"{'z [mm]':>8}  {'t_arrival':>14}  {'Peak Phi [W/m²]':>16}  {'Peak T [HeV]':>14}")
    print("-" * 72)
    for ds in datasets:
        t_arr = ds['t_arrival']
        t_str = f"{t_arr:.4g}" if t_arr is not None else "N/A"
        peak_flux = np.max(ds['flux_raw'])
        peak_T = np.max(ds['T_hev'])
        print(f"{ds['z_pos_mm']:8.1f}  {t_str:>14}  {peak_flux:16.4e}  {peak_T:14.4f}")
    print("=" * 72 + "\n")


# --------------------------------------------------------------------------
# Convenience: run from the Marshak march and plot
# --------------------------------------------------------------------------

def compute_and_plot_radiation_flux(
    times_to_store,
    *,
    mode="marshak_wall_loss",
    wall_material="Gold",
    use_seconds=True,
    vary_rho=False,
    lam_eff=False,
    power=2,
    detector_positions_mm=None,
    save_csv=True,
    show_plot=True,
):
    """
    End-to-end convenience function:
      1. Run the Marshak march to obtain z_F(t) and Ts(t).
      2. Compute T(z,t) via the Henyey profile at each detector.
      3. Compute Φ(t) = σ_SB · T⁴.
      4. Normalise and plot.

    Parameters
    ----------
    times_to_store : array-like
        Time points [ns or seconds depending on *use_seconds*].
    mode : str
        Solver mode passed to *analytic_wave_front_dispatch*.
    wall_material : str
        Wall material for the solver.
    detector_positions_mm : list of float, optional
        Detector locations [mm].  Default [0.5, 1.0, 1.5].
    save_csv : bool
        If True, save results to a CSV in Data_new/<Experiment>/<Material>/.
    show_plot : bool
        If True, display a matplotlib figure.

    Returns
    -------
    datasets : list of dict
        Flux datasets for each detector position.
    """
    # --- Step 1: obtain z_F(t) and Ts(t) from the wave-front solver ---
    result = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=use_seconds,
        mode=mode,
        wall_material=wall_material,
        vary_rho=vary_rho,
        lam_eff=lam_eff,
        power=power,
    )
    xF = result[0]       # heat-front position [cm]
    Ts = result[1]       # surface temperature [HeV]

    times = np.asarray(times_to_store, dtype=float)

    # --- Step 2–3: flux computation + normalisation ---
    datasets = compute_radiation_flux_datasets(
        times, xF, Ts,
        detector_positions_mm=detector_positions_mm,
    )

    # --- Print summary ---
    print_flux_summary(datasets)

    # --- Step 4a: save to CSV ---
    if save_csv:
        out_dir = BASE_DIR / "Data_new" / Experiment / Material / "1.5 model"/ "radiation_flux"
        out_dir.mkdir(parents=True, exist_ok=True)
        columns = {"time_ns": times}
        for ds in datasets:
            tag = f"z{ds['z_pos_mm']:.1f}mm"
            columns[f"T_hev_{tag}"] = ds['T_hev']
            columns[f"flux_raw_{tag}"] = ds['flux_raw']
            columns[f"flux_norm_{tag}"] = ds['flux_normalised']
        df = pd.DataFrame(columns)
        csv_path = out_dir / f"radiation_flux_{mode}_{wall_material}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved flux data -> {csv_path}")

    # --- Step 4b: plot ---
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        colors = ['#e6194b', '#3cb44b', '#4363d8']

        # Panel 1: raw flux
        ax = axes[0]
        for ds, c in zip(datasets, colors):
            label = f"z = {ds['z_pos_mm']:.1f} mm"
            ax.plot(ds['times'], ds['flux_raw'], color=c, linewidth=1.5, label=label)
            if ds['t_arrival'] is not None:
                ax.axvline(ds['t_arrival'], color=c, linestyle=':', alpha=0.5)
        ax.set_xlabel("Time (ns)", fontsize=14, fontname='serif')
        ax.set_ylabel(r"$\Phi$ [W/m²]", fontsize=14, fontname='serif')
        ax.set_title("Raw Radiation Flux", fontsize=15, fontname='serif')
        ax.legend(prop={'family': 'serif'})
        ax.grid(True, alpha=0.3)

        # Panel 2: normalised flux
        ax = axes[1]
        for ds, c in zip(datasets, colors):
            label = f"z = {ds['z_pos_mm']:.1f} mm"
            ax.plot(ds['times'], ds['flux_normalised'], color=c, linewidth=1.5, label=label)
            if ds['t_arrival'] is not None:
                ax.axvline(ds['t_arrival'], color=c, linestyle=':', alpha=0.5)
        ax.set_xlabel("Time (ns)", fontsize=14, fontname='serif')
        ax.set_ylabel(r"$\Phi / \Phi_{\max}$", fontsize=14, fontname='serif')
        ax.set_title("Normalised Radiation Flux", fontsize=15, fontname='serif')
        ax.legend(prop={'family': 'serif'})
        ax.grid(True, alpha=0.3)

        # Panel 3: temperature at each detector
        ax = axes[2]
        for ds, c in zip(datasets, colors):
            label = f"z = {ds['z_pos_mm']:.1f} mm"
            ax.plot(ds['times'], ds['T_hev'], color=c, linewidth=1.5, label=label)
            if ds['t_arrival'] is not None:
                ax.axvline(ds['t_arrival'], color=c, linestyle=':', alpha=0.5)
        ax.set_xlabel("Time (ns)", fontsize=14, fontname='serif')
        ax.set_ylabel("T (HeV)", fontsize=14, fontname='serif')
        ax.set_title("Temperature at Detectors", fontsize=15, fontname='serif')
        ax.legend(prop={'family': 'serif'})
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Radiation Flux Analysis — {Material}  ({mode}, wall={wall_material})",
            fontsize=16, fontname='serif', y=1.02,
        )
        plt.tight_layout()

        save_figure(f"radiation_flux_{mode}_{wall_material}.png", model1_5=True)

    return datasets


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("STARTING STANDALONE RADIATION FLUX COMPUTATION")
    print("=" * 72)
    
    print(f"Active Material: {Material}")
    
    # Select default time array based on Material (consistent with comparison.py)
    if Material == "SiO2":
        times = np.linspace(0.01, 3.0, 1000)
    elif Material == "SiO2_low_energy":
        times = np.linspace(0.01, 15.0, 1000)
    elif Material == "C11H16Pb0.3852":
        times = np.linspace(0.01, 1.0, 1000)
    elif Material in ["C6H12", "C6H12Cu0.394"]:
        times = np.linspace(0.01, 2.0, 1000)
    elif Material == "Ta2O5":
        times = np.linspace(0.01, 3.0, 1000)
    elif Material == "SiO2_Moore":
        times = np.linspace(0.01, 4.0, 1000)
    elif Material == "C8H7Cl":
        times = np.linspace(0.01, 4.0, 1000)
    elif Material in ["C15H20O6", "C15H20O6Au0.172"]:
        times = np.linspace(0.01, 3.0, 1000)
    elif Material == "C8H8":
        times = np.linspace(0.01, 1.5, 1000)
    elif Material == "french_gold":
        times = np.linspace(0.01, 4.0, 1000)
    elif Material == "french_cupper":
        times = np.linspace(0.01, 4.0, 1000)
    else:
        print("Unknown material. Defaulting to times [0.01, 3.0] ns.")
        times = np.linspace(0.01, 3.0, 1000)

    # Compute radiation flux and generate plots
    compute_and_plot_radiation_flux(
        times,
        mode="marshak_ablation",
        vary_rho = False,
        wall_material="Gold",
        show_plot=True,
        save_csv=True,
    )
    print("Done! CSV results saved under Data_new/ and plot saved under Figures_new/.")

