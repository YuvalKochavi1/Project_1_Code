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

# Font configuration: serif and LaTeX style math font
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'axes.unicode_minus': False,
})
from parameters import alpha, beta, K_per_Hev, Experiment, Material, a_hev, c, r_grid as R_GRID_DEFAULT, alpha_gold, beta_gold, alpha_copper, beta_copper, alpha_be, beta_be, R_cm, z, L
from model_main import analytic_wave_front_dispatch, BASE_DIR
from csv_helpers import save_figure, save_series_csv
from shape_2D_analytical_model import _compute_wall_heyney_horizontal_profile, _compute_temperature_mesh
from scipy import special

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
    import pandas as pd
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

    # --- Load experimental data ---
    exp_data = {}
    if Material == "SiO2_low_energy":
        for d in [0.5, 1.0, 1.5]:
            csv_file = f"{d:g}mm.csv"
            if d == 1.0:
                csv_file = "1mm.csv"
            csv_path = BASE_DIR / "Data_new" / "Back" / "SiO2_low_energy" / "article" / "flux" / csv_file
            if csv_path.exists():
                exp_data[d] = pd.read_csv(csv_path)

    # --- Step 4b: plot ---
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

        # Panel 1: raw flux
        ax = axes[0]
        for ds, c in zip(datasets, colors):
            label = fr"$z = {ds['z_pos_mm']:g}$ mm"
            ax.plot(ds['times'], ds['flux_raw'], color=c, linewidth=1.5, label=label)
            if ds['t_arrival'] is not None:
                ax.axvline(ds['t_arrival'], color=c, linestyle=':', alpha=0.5)
            
            # Plot experimental data
            z_mm = float(ds['z_pos_mm'])
            if z_mm in exp_data:
                # Plot as is without normalizing
                df_exp = exp_data[z_mm]
                ax.plot(df_exp['x'], df_exp['y'], color=c, linestyle='--', linewidth=2.0, label=fr"Data ${z_mm:g}$ mm")

        ax.set_xlabel(r"$t$ [ns]", fontsize=14, fontname='serif')
        ax.set_ylabel(r"$\Phi$ [$\mathrm{W/m}^2$]", fontsize=14, fontname='serif')
        ax.set_title("Raw Radiation Flux", fontsize=15, fontname='serif')
        ax.legend(prop={'family': 'serif'})
        ax.grid(True, alpha=0.3)

        # Panel 2: normalised flux
        ax = axes[1]
        for ds, c in zip(datasets, colors):
            label = fr"$z = {ds['z_pos_mm']:g}$ mm"
            ax.plot(ds['times'], ds['flux_normalised'], color=c, linewidth=1.5, label=label)
            if ds['t_arrival'] is not None:
                ax.axvline(ds['t_arrival'], color=c, linestyle=':', alpha=0.5)
            
            # Plot experimental data here too just in case
            z_mm = float(ds['z_pos_mm'])
            if z_mm in exp_data:
                df_exp = exp_data[z_mm]
                ax.plot(df_exp['x'], df_exp['y'], color=c, linestyle='--', linewidth=2.0, label=fr"Data ${z_mm:g}$ mm")

        ax.set_xlabel(r"$t$ [ns]", fontsize=14, fontname='serif')
        ax.set_ylabel(r"$\Phi / \Phi_{\max}$", fontsize=14, fontname='serif')
        ax.set_title("Normalised Radiation Flux", fontsize=15, fontname='serif')
        ax.legend(prop={'family': 'serif'})
        ax.grid(True, alpha=0.3)

        # Panel 3: temperature at each detector
        ax = axes[2]
        for ds, c in zip(datasets, colors):
            label = fr"$z = {ds['z_pos_mm']:g}$ mm"
            ax.plot(ds['times'], ds['T_hev'], color=c, linewidth=1.5, label=label)
            if ds['t_arrival'] is not None:
                ax.axvline(ds['t_arrival'], color=c, linestyle=':', alpha=0.5)
        ax.set_xlabel(r"$t$ [ns]", fontsize=14, fontname='serif')
        ax.set_ylabel(r"$T$ [heV]", fontsize=14, fontname='serif')
        ax.set_title("Temperature at Detectors", fontsize=15, fontname='serif')
        ax.legend(prop={'family': 'serif'})
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Radiation Flux Analysis - {Material} ({mode}, wall={wall_material})",
            fontsize=16, fontname='serif', y=1.02,
        )
        plt.tight_layout()

        save_figure(f"radiation_flux_{mode}_{wall_material}.png", model1_5=True)

    return datasets


# ==========================================================================
# 2D FLUX CURVATURE: Radial flux profile Φ(r) at detector positions
# ==========================================================================
#
# These functions compute the radial flux cross-section at each detector
# z-position using the curved front z_F(r,t) = z_F(0,t)·J₀(κ₀·r).
#
# flux at a given depth — which can be compared to Back's experimental
# measurements.
# ==========================================================================


def compute_temperature_cross_section_at_z(
    z_pos_cm, r_grid, z_F_radial, Ts_hev, alpha_val=None, beta_val=None,
    snapshot=None, wall_material="Gold", ablation=True
):
    """
    Compute the radial temperature profile T(r) at a fixed axial depth
    *z_pos_cm*, given the radially-varying front position *z_F_radial*.
    Extends the profile into the wall (gold) region if snapshot contains
    the wall information.

    Uses the Henyey self-similar profile:
        T(r) = Ts · (1 − z_pos / z_F(r))^{1/(4+α−β)}
    where z_F(r) is the front position at each radial coordinate.
    Returns 0 where z_pos >= z_F(r) (ahead of the front).

    Parameters
    ----------
    z_pos_cm : float
        Axial detector depth [cm].
    r_grid : 1-D array
        Radial grid [cm].
    z_F_radial : 1-D array
        Front position z_F(r) for each radial point [cm].
    Ts_hev : float
        Surface temperature at this time [HeV].
    alpha_val, beta_val : float, optional
        Opacity / EOS exponents.  Default to globals.
    snapshot : dict, optional
        Contains grid, mask, and penetration profile data for the wall.
    wall_material : str
        Wall material (e.g., 'Gold', 'Copper', 'Be', 'Vacuum').
    ablation : bool
        Whether to use moving ablation boundary / wall mask.

    Returns
    -------
    T_radial : 1-D array
        Temperature [HeV] at each radial position at depth z_pos_cm.
    """
    if alpha_val is None:
        alpha_val = alpha
    if beta_val is None:
        beta_val = beta

    r_grid = np.asarray(r_grid, dtype=float)
    z_F_radial = np.asarray(z_F_radial, dtype=float)

    exponent = 1.0 / (4.0 + alpha_val - beta_val)

    # Where the front has passed the detector depth in foam
    behind_front = (z_F_radial > z_pos_cm) & (z_F_radial > 0)

    ratio = np.where(behind_front, 1.0 - z_pos_cm / z_F_radial, 0.0)
    ratio = np.clip(ratio, 0.0, None)

    T_foam = np.where(behind_front, Ts_hev * (ratio ** exponent), 0.0)

    # If wall details are not available or not requested, return only the foam profile.
    if snapshot is None or 'r_gold_grid' not in snapshot:
        return T_foam

    r_mesh_foam = snapshot.get('r_grid', r_grid)
    r_mesh_wall = snapshot.get('r_gold_grid')
    if r_mesh_wall is None:
        return T_foam

    # Find closest z-index in z_grid
    z_grid = snapshot.get('z_grid')
    if z_grid is not None:
        idx_z = int(np.argmin(np.abs(z_grid - z_pos_cm)))
    else:
        idx_z = 0

    # Extract 1D profiles shaped as 2D (1, N) for _compute_wall_heyney_horizontal_profile
    foam_mask_2d = snapshot.get('ablation_foam_mask')
    foam_mask_1d = foam_mask_2d[idx_z : idx_z + 1] if foam_mask_2d is not None else None

    wall_mask_2d = snapshot.get('ablation_wall_mask')
    wall_mask_1d = wall_mask_2d[idx_z : idx_z + 1] if wall_mask_2d is not None else None

    pen_profile_2d = snapshot.get('wall_penetration_radius_profile')
    pen_profile_1d = pen_profile_2d[idx_z : idx_z + 1] if pen_profile_2d is not None else None

    shock_profile_2d = snapshot.get('shock_penetration_radius_profile')
    shock_profile_1d = shock_profile_2d[idx_z : idx_z + 1] if shock_profile_2d is not None else None

    # Resolve wall exponent
    if wall_material == "Gold":
        exponent_wall = 1.0 / (4.0 + alpha_gold - beta_gold)
    elif wall_material == "Copper":
        exponent_wall = 1.0 / (4.0 + alpha_copper - beta_copper)
    elif wall_material == "Be":
        exponent_wall = 1.0 / (4.0 + alpha_be - beta_be)
    elif wall_material == "Vacuum":
        exponent_wall = 0.0
    else:
        exponent_wall = exponent

    # Call horizontal wall profile builder
    T_foam_2d = T_foam[np.newaxis, :] # (1, Nr_foam)
    T_wall_profile = _compute_wall_heyney_horizontal_profile(
        T_foam_2d,
        foam_mask_1d,
        wall_mask_1d,
        r_mesh_foam,
        exponent_wall,
        is_ablation=ablation,
        r_mesh_wall=r_mesh_wall,
        penetration_radius_profile=pen_profile_1d,
        shock_radius_profile=shock_profile_1d,
    )
    T_wall_1d = T_wall_profile[0]

    # Combine foam and wall profiles
    T_mesh_plot = np.zeros(r_mesh_wall.size, dtype=float)
    foam_domain = r_mesh_wall <= R_cm
    T_mesh_plot[foam_domain] = np.interp(
        r_mesh_wall[foam_domain],
        r_mesh_foam,
        T_foam,
        left=0.0,
        right=0.0,
    )

    wall_valid = np.isfinite(T_wall_1d)
    if np.any(wall_valid):
        T_mesh_plot[wall_valid] = T_wall_1d[wall_valid]

    # Convert NaNs to 0.0 to make sure flux calculations and plotting are safe
    T_radial = np.nan_to_num(T_mesh_plot, nan=0.0)
    return T_radial


def compute_flux_curvature_at_position(
    z_pos_cm, r_grid, z_F_radial, Ts_hev, alpha_val=None, beta_val=None,
    snapshot=None, wall_material="Gold", ablation=True
):
    """
    Compute the radial radiation flux profile Φ(r) = σ_SB · T(r)⁴ at a
    fixed axial depth *z_pos_cm*.

    Parameters
    ----------
    z_pos_cm : float
        Detector depth [cm].
    r_grid : 1-D array
        Radial grid [cm].
    z_F_radial : 1-D array
        Front position z_F(r) at each radial point [cm].
    Ts_hev : float
        Surface temperature [HeV].
    alpha_val, beta_val : float, optional
        Opacity / EOS exponents.
    snapshot : dict, optional
        Wall bessel snapshot.
    wall_material : str
        Wall material.
    ablation : bool
        Whether using ablation mode.

    Returns
    -------
    dict with keys:
        'r_grid'         – radial grid [cm]
        'z_pos_cm'       – detector depth [cm]
        'z_pos_mm'       – detector depth [mm]
        'T_radial_hev'   – temperature profile T(r) [HeV]
        'flux_radial'    – radiation flux Φ(r) [W/m²]
    """
    T_radial = compute_temperature_cross_section_at_z(
        z_pos_cm, r_grid, z_F_radial, Ts_hev,
        alpha_val=alpha_val, beta_val=beta_val,
        snapshot=snapshot, wall_material=wall_material, ablation=ablation
    )
    
    r_out = r_grid
    if snapshot is not None and 'r_gold_grid' in snapshot:
        r_out = snapshot['r_gold_grid']
        
    flux_radial = SIGMA_SB * T_radial ** 4

    return {
        'r_grid':        np.asarray(r_out, dtype=float),
        'z_pos_cm':      z_pos_cm,
        'z_pos_mm':      z_pos_cm * 10.0,
        'T_radial_hev':  T_radial,
        'flux_radial':   flux_radial,
    }


def compute_flux_curvature_datasets(
    bessel_data,
    z_F_array,
    Ts_array,
    times_array,
    times_ns_snapshots,
    *,
    detector_positions_mm=None,
    alpha_val=None,
    beta_val=None,
    wall_material="Gold",
    ablation=True,
):
    """
    Build flux curvature datasets for multiple time snapshots and
    detector positions, extending into the wall (gold) region.
    """
    if detector_positions_mm is None:
        detector_positions_mm = DETECTOR_POSITIONS_MM
    if alpha_val is None:
        alpha_val = alpha
    if beta_val is None:
        beta_val = beta

    times_array = np.asarray(times_array, dtype=float)

    results = {}
    all_peaks = []

    for t_target in times_ns_snapshots:
        t_target = float(t_target)

        # --- Find closest available time in bessel_data ---
        if bessel_data and len(bessel_data) > 0:
            available = np.array(list(bessel_data.keys()), dtype=float)
            t_closest = float(available[np.argmin(np.abs(available - t_target))])
            snapshot = bessel_data[t_closest]
            r_grid_snap = np.asarray(snapshot['r_grid'], dtype=float)
            z_F_radial = np.asarray(snapshot['z_F_radial'], dtype=float)
        else:
            # Fallback: flat front (no radial variation)
            t_closest = t_target
            snapshot = None
            r_grid_snap = np.asarray(R_GRID_DEFAULT, dtype=float)
            z_F_t = float(np.interp(t_closest, times_array, z_F_array))
            z_F_radial = np.full_like(r_grid_snap, z_F_t)

        # Interpolate Ts at this time
        Ts_t = float(np.interp(t_closest, times_array, Ts_array))

        results[t_closest] = {}

        for z_mm in detector_positions_mm:
            z_cm = z_mm / 10.0
            ds = compute_flux_curvature_at_position(
                z_cm, r_grid_snap, z_F_radial, Ts_t,
                alpha_val=alpha_val, beta_val=beta_val,
                snapshot=snapshot, wall_material=wall_material, ablation=ablation
            )
            ds['t_ns'] = t_closest
            ds['kappa_0'] = snapshot['kappa_0'] if (snapshot is not None and 'kappa_0' in snapshot) else None
            results[t_closest][z_mm] = ds

            peak = float(np.max(ds['flux_radial']))
            if peak > 0:
                all_peaks.append(peak)

    # --- Normalisation ---
    global_peak = max(all_peaks) if all_peaks else 1.0
    results['global_peak'] = global_peak

    for t_ns_key in list(results.keys()):
        if t_ns_key == 'global_peak':
            continue
        for z_mm_key in results[t_ns_key]:
            ds = results[t_ns_key][z_mm_key]
            peak = float(np.max(ds['flux_radial']))
            if peak > 0:
                ds['flux_normalised'] = ds['flux_radial'] / peak
            else:
                ds['flux_normalised'] = np.zeros_like(ds['flux_radial'])

    return results


def plot_flux_curvature(
    curvature_datasets,
    *,
    detector_positions_mm=None,
    show_plot=True,
    save_csv=True,
    title_suffix="",
):
    """
    Plot radial flux profiles (flux curvature) for each time snapshot
    and detector position.
    """
    if detector_positions_mm is None:
        detector_positions_mm = DETECTOR_POSITIONS_MM

    global_peak = curvature_datasets.get('global_peak', 1.0)

    # Collect time keys (skip metadata keys)
    time_keys = sorted(
        k for k in curvature_datasets.keys() if k != 'global_peak'
    )

    n_times = len(time_keys)
    n_detectors = len(detector_positions_mm)

    if n_times == 0 or n_detectors == 0:
        print("Warning: no data to plot for flux curvature.")
        return

    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4']

    fig, axes = plt.subplots(
        n_detectors, n_times,
        figsize=(6 * n_times, 4.5 * n_detectors),
        squeeze=False,
    )

    for j, t_ns in enumerate(time_keys):
        for i, z_mm in enumerate(detector_positions_mm):
            ax = axes[i][j]
            if z_mm not in curvature_datasets[t_ns]:
                ax.set_visible(False)
                continue

            ds = curvature_datasets[t_ns][z_mm]
            r_mm = ds['r_grid'] * 10.0  # cm → mm

            color = colors[i % len(colors)]

            # --- Symmetric Normalised flux ---
            r_sym = np.concatenate((-r_mm[::-1], r_mm[1:]))
            flux_sym = np.concatenate((ds['flux_normalised'][::-1], ds['flux_normalised'][1:]))
            ax.plot(
                r_sym, flux_sym,
                color=color, linewidth=2.0,
                label=fr"$z = {z_mm:.1f}$ mm",
            )

            # Load experimental data if available and we are at z = 1.0 mm and t approx 9.5 ns
            if Material == "SiO2_low_energy" and abs(z_mm - 1.0) < 1e-3 and abs(t_ns - 9.5) < 0.2:
                csv_path = BASE_DIR / "Data_new" / "Back" / "SiO2_low_energy" / "article" / "flux_curvature" / "experimental_results.csv"
                if csv_path.exists():
                    try:
                        df_exp = pd.read_csv(csv_path)
                        x_exp = df_exp['x'].to_numpy() # radius in mm
                        y_exp = df_exp['y'].to_numpy() # flux values
                        # Normalize according to the point with the closest x-value to zero
                        closest_idx = np.argmin(np.abs(x_exp))
                        y_anchor = y_exp[closest_idx]
                        if y_anchor > 0:
                            y_exp_norm = y_exp / y_anchor
                        else:
                            y_exp_norm = y_exp
                        
                        ax.scatter(
                            x_exp, y_exp_norm,
                            color='black', marker='o', facecolors='none', s=45, linewidths=1.5,
                            label='Experiment (Back et al.)', zorder=5
                        )
                    except Exception as e:
                        print(f"Error loading experimental results: {e}")

            # Plot the corresponding Bessel function J0(kappa_0 * r)^4 symmetrically for flux curvature comparison
            # kappa_val = ds.get('kappa_0')
            # if kappa_val is not None:
            #     # Radial coordinates within the foam region (r_mm <= R_cm * 10.0)
            #     foam_mask_r = r_mm <= R_cm * 10.0
            #     r_foam_mm = r_mm[foam_mask_r]
            #     r_foam_cm = r_foam_mm / 10.0
            #     j0_foam_4th = special.j0(kappa_val * r_foam_cm) ** 4
                
            #     # Plot symmetrically
            #     r_foam_sym = np.concatenate((-r_foam_mm[::-1], r_foam_mm[1:]))
            #     j0_foam_sym = np.concatenate((j0_foam_4th[::-1], j0_foam_4th[1:]))
                
            #     ax.plot(
            #         r_foam_sym, j0_foam_sym,
            #         color='blue', linestyle=':', linewidth=2.0,
            #         label=r'$J_0(\kappa_0 r)^4$'
            #     )

            # Draw original Foam-Wall interface symmetrically on both sides
            # ax.axvline(
            #     x=R_cm * 10.0,
            #     color='gray', linestyle='--', alpha=0.7,
            #     label='Foam-Wall interface'
            # )
            # ax.axvline(
            #     x=-R_cm * 10.0,
            #     color='gray', linestyle='--', alpha=0.7
            # )

            ax.set_xlabel(r"$r$ [mm]", fontsize=12, fontname='serif')
            if j == 0:
                ax.set_ylabel(
                    r"Flux [a.u]",
                    fontsize=12, fontname='serif',
                )

            ax.set_xlim(-r_mm[-1], r_mm[-1])
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(prop={'family': 'serif'}, fontsize=10)
    plt.tight_layout()
    save_figure(
        f"flux_curvature{title_suffix.replace(' ', '_')}.png",
        model1_5=False, model2_D=True
    )

    # --- CSV export ---
    if save_csv:
        out_dir = (
            BASE_DIR / "Data_new" / Experiment / Material
            / "2D_shape" / "flux_curvature"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        for t_ns in time_keys:
            columns = {}
            first_key = list(curvature_datasets[t_ns].keys())[0]
            columns["r_cm"] = curvature_datasets[t_ns][first_key]['r_grid']
            columns["r_mm"] = columns["r_cm"] * 10.0

            for z_mm in detector_positions_mm:
                if z_mm not in curvature_datasets[t_ns]:
                    continue
                ds = curvature_datasets[t_ns][z_mm]
                tag = f"z{z_mm:.1f}mm"
                columns[f"T_hev_{tag}"] = ds['T_radial_hev']
                columns[f"flux_raw_{tag}"] = ds['flux_radial']
                columns[f"flux_norm_{tag}"] = ds['flux_normalised']

            csv_path = out_dir / f"flux_curvature_t{t_ns:.2f}ns.csv"
            save_series_csv(csv_path, columns)
            print(f"Saved flux curvature CSV -> {csv_path}")

    return fig


def plot_flux_curvature_post_breakout(
    times_to_store,
    *,
    mode="marshak_ablation",
    wall_material="Gold",
    use_seconds=True,
    vary_rho=True,
    lam_eff=True,
    power=1,
    detector_positions_mm=None,
    delay_ns=0.5,
    show_plot=True,
):
    """
    For the active material, plot the radial flux curvature at a fixed delay (e.g. 0.5 ns)
    after the breakout time for each detector position, all on the same graph, using global
    normalization to show relative attenuation.
    """
    if detector_positions_mm is None:
        if Material == "Ta2O5":
            detector_positions_mm = [0.25, 0.5, 0.75, 1.0]
        else:
            detector_positions_mm = [0.5, 1.0, 1.5]

    # Run the solver
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
    bessel_data = result[5] if len(result) > 5 else {}
    times = np.asarray(times_to_store, dtype=float)

    if not bessel_data:
        print("Error: No bessel_data available in solver. Cannot compute curvature.")
        return

    ablation = "ablation" in mode
    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']

    # Pre-collect computed profiles so we can find global peak
    raw_profiles = []
    for idx, z_mm in enumerate(detector_positions_mm):
        z_cm = z_mm / 10.0
        
        # Detect arrival/breakout time
        _, t_breakout = detect_arrival_time(z_cm, times, xF)
        if t_breakout is None:
            print(f"Heat front never reaches z = {z_mm:.1f} mm. Skipping.")
            continue
        if (z_mm == 0.25 or z_mm == 0.5) and Material == "Ta2O5": 
            t_target = t_breakout + delay_ns - 0.1
        else:
            t_target = t_breakout + delay_ns
        #t_target = t_breakout + delay_ns
        
        # Find closest snapshot in bessel_data
        available = np.array(list(bessel_data.keys()), dtype=float)
        t_closest = float(available[np.argmin(np.abs(available - t_target))])
        
        snapshot = bessel_data[t_closest]
        r_grid_snap = np.asarray(snapshot['r_grid'], dtype=float)
        z_F_radial = np.asarray(snapshot['z_F_radial'], dtype=float)
        Ts_t = float(np.interp(t_closest, times, Ts))
        
        # Compute curvature at this depth and time
        ds = compute_flux_curvature_at_position(
            z_cm, r_grid_snap, z_F_radial, Ts_t,
            snapshot=snapshot, wall_material=wall_material, ablation=ablation
        )
        
        raw_profiles.append((z_mm, t_breakout, t_closest, ds))

    if not raw_profiles:
        print("Error: No profiles could be computed.")
        return

    # Find the global maximum peak across all calculated profiles
    global_peak = max(float(np.max(p[3]['flux_radial'])) for p in raw_profiles)
    if global_peak <= 0:
        global_peak = 1.0

    plt.figure(figsize=(8, 6))

    # --- Pre-load experimental Data to calculate global normalisation factor (Ta2O5 / SiO2)
    exp_ta2o5 = {}
    y_norm = 1.0
    exp_sio2 = {}
    exp_sio2_ta2o5 = {}
    exp_sio2_model_ta2o5 = {}
    if Material == "Ta2O5":
        base_ta_path = BASE_DIR / "Data_new" / "Back" / "Ta2O5" / "article" / "flux_curvature"
        # files 1.csv -> 0.25mm, 2.csv -> 0.5mm, 3.csv -> 0.75mm, 4.csv -> 1.0mm
        file_mapping = {0.25: "1new.csv", 0.5: "2new.csv", 0.75: "3new.csv", 1.0: "4new.csv"}
        for z_key, fname in file_mapping.items():
            csv_path = base_ta_path / fname
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    exp_ta2o5[z_key] = df
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        
        # Normalize the experimental data according to "1new.csv" at x closest to 0
        if 0.25 in exp_ta2o5:
            df_ref = exp_ta2o5[0.25]
            idx_closest = df_ref['x'].abs().idxmin()
            y_norm = df_ref.loc[idx_closest, 'y']
            print(f"Ta2O5 normalization factor from 1new.csv at x ~ 0: {y_norm}")
    elif Material == "SiO2":
        base_sio2_path = BASE_DIR / "Data_new" / "Back" / "SiO2" / "article" / "flux"
        csv_path = base_sio2_path / "1mm.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                exp_sio2[1.0] = df
            except Exception as e:
                print(f"Error loading SiO2 1mm.csv: {e}")
        
        csv_ta2o5_path = base_sio2_path / "Ta2O5_1mm.csv"
        if csv_ta2o5_path.exists():
            try:
                df_ta = pd.read_csv(csv_ta2o5_path)
                exp_sio2_ta2o5[1.0] = df_ta
            except Exception as e:
                print(f"Error loading SiO2 Ta2O5_1mm.csv: {e}")

        csv_model_ta2o5_path = base_sio2_path / "modelTa2O5.csv"
        if csv_model_ta2o5_path.exists():
            try:
                df_mta = pd.read_csv(csv_model_ta2o5_path)
                exp_sio2_model_ta2o5[1.0] = df_mta
            except Exception as e:
                print(f"Error loading SiO2 modelTa2O5.csv: {e}")

    r_mm_last = None
    for idx, (z_mm, t_breakout, t_closest, ds) in enumerate(raw_profiles):
        r_mm = ds['r_grid'] * 10.0
        r_mm_last = r_mm
        flux_raw = ds['flux_radial']
        
        # Global normalization (relative scaling)
        flux_norm = flux_raw / global_peak
            
        # Symmetric profiles
        r_sym = np.concatenate((-r_mm[::-1], r_mm[1:]))
        flux_sym = np.concatenate((flux_norm[::-1], flux_norm[1:]))
        
        label = fr"$z = {z_mm:.2f}$ mm"
        plt.plot(r_sym, flux_sym, color=colors[idx % len(colors)], linewidth=2.2, label=label)

        # Plot Ta2O5 experimental data
        if Material == "Ta2O5":
            for z_key, df_exp in exp_ta2o5.items():
                if abs(z_mm - z_key) < 1e-3:
                    x_exp = df_exp['x'].to_numpy()
                    y_exp = df_exp['y'].to_numpy()
                    if y_norm > 0:
                        y_exp_norm = y_exp / y_norm
                    else:
                        y_exp_norm = y_exp
                    plt.plot(
                        x_exp, y_exp_norm,
                        color=colors[idx % len(colors)], linestyle='--', linewidth=2.0,
                        # No label so it doesn't clutter the legend or set a simple one
                        zorder=5
                    )
                    break
        # Plot SiO2 experimental data
        elif Material == "SiO2":
            for z_key, df_exp in exp_sio2.items():
                if abs(z_mm - z_key) < 1e-3:
                    x_exp = df_exp['x'].to_numpy()
                    y_exp = df_exp['y'].to_numpy()
                    y_exp_norm = y_exp / np.max(y_exp) if np.max(y_exp) > 0 else y_exp
                    
                    # Symmetric mapping
                    x_sym = np.concatenate((-x_exp[::-1], x_exp[1:]))
                    y_sym = np.concatenate((y_exp_norm[::-1], y_exp_norm[1:]))
                    
                    plt.plot(
                        x_sym, y_sym,
                        color=colors[idx % len(colors)], linestyle='--', linewidth=2.0,
                        label="Experiment (1mm.csv)",
                        zorder=5
                    )
                    break

            for z_key, df_exp in exp_sio2_ta2o5.items():
                if abs(z_mm - z_key) < 1e-3:
                    x_exp = df_exp['x'].to_numpy()
                    y_exp = df_exp['y'].to_numpy()
                    y_exp_norm = y_exp / np.max(y_exp) if np.max(y_exp) > 0 else y_exp
                    
                    # Symmetric mapping
                    x_sym = np.concatenate((-x_exp[::-1], x_exp[1:]))
                    y_sym = np.concatenate((y_exp_norm[::-1], y_exp_norm[1:]))
                    
                    plt.plot(
                        x_sym, y_sym,
                        color='blue', linestyle=':', linewidth=2.0,
                        label="Experiment (Ta2O5 1mm.csv)",
                        zorder=5
                    )
                    break

            for z_key, df_exp in exp_sio2_model_ta2o5.items():
                if abs(z_mm - z_key) < 1e-3:
                    x_exp = df_exp['x'].to_numpy()
                    y_exp = df_exp['y'].to_numpy()
                    y_exp_norm = y_exp / np.max(y_exp) if np.max(y_exp) > 0 else y_exp
                    
                    # Already spans both positive and negative x
                    x_sym = x_exp
                    y_sym = y_exp_norm
                    
                    plt.plot(
                        x_sym, y_sym,
                        color='blue', linestyle='-.', linewidth=2.0,
                        label="Model Ta2O5 (modelTa2O5.csv)",
                        zorder=5
                    )
                    break


    # Draw original Foam-Wall interface symmetrically
    # plt.axvline(x=R_cm * 10.0, color='gray', linestyle='--', alpha=0.7, label='Foam-Wall interface')
    # plt.axvline(x=-R_cm * 10.0, color='gray', linestyle='--')

    plt.xlabel(r"$r$ [mm]", fontsize=13, fontname='serif')
    plt.ylabel(r"Flux [a.u.]", fontsize=13, fontname='serif')
    if r_mm_last is not None:
        plt.xlim(-r_mm_last[-1], r_mm_last[-1])
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend(prop={'family': 'serif'}, loc='upper left', fontsize=10)
    plt.tight_layout()

    save_figure(f"flux_curvature_post_breakout_{delay_ns}ns.png", model1_5=False, model2_D=True)
    if show_plot:
        plt.show()


def compute_and_plot_flux_curvature(
    times_to_store,
    *,
    times_ns_snapshots=None,
    mode="marshak_ablation",
    wall_material="Gold",
    use_seconds=True,
    vary_rho=True,
    lam_eff=False,
    power=1.5,
    detector_positions_mm=None,
    save_csv=True,
    show_plot=True,
    title_suffix="",
):
    """
    End-to-end convenience function for flux curvature analysis:
      1. Run the Marshak solver to get z_F(t), Ts(t), and bessel_data.
      2. For selected time snapshots, extract the curved front z_F(r).
      3. At each detector depth, compute the radial temperature
         cross-section T(r, z_det, t) and the flux Φ(r) = σ_SB·T⁴.
      4. Normalise, plot, and save.
    """
    if times_ns_snapshots is None:
        times_ns_snapshots = [1.0, 2.0, 2.5]

    # --- Step 1: run the solver ---
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
    bessel_data = result[5] if len(result) > 5 else {}

    times = np.asarray(times_to_store, dtype=float)

    if not bessel_data:
        print(
            "Warning: solver mode did not return bessel_data. "
            "Radial profiles will be flat (no curvature)."
        )

    # --- Step 2–3: flux curvature computation ---
    ablation = "ablation" in mode
    curvature_datasets = compute_flux_curvature_datasets(
        bessel_data,
        xF, Ts, times,
        times_ns_snapshots,
        detector_positions_mm=detector_positions_mm,
        wall_material=wall_material,
        ablation=ablation,
    )

    # --- Print summary ---
    print("\n" + "=" * 72)
    print("FLUX CURVATURE SUMMARY")
    print("-" * 72)
    print(f"{'t [ns]':>8}  {'z [mm]':>8}  {'Peak Phi':>16}  {'Center T [HeV]':>14}")
    print("-" * 72)
    for t_ns in sorted(
        k for k in curvature_datasets.keys() if k != 'global_peak'
    ):
        for z_mm in sorted(curvature_datasets[t_ns].keys()):
            ds = curvature_datasets[t_ns][z_mm]
            peak_flux = float(np.max(ds['flux_radial']))
            center_T = float(ds['T_radial_hev'][0])
            print(
                f"{t_ns:8.2f}  {z_mm:8.1f}  {peak_flux:16.4e}  {center_T:14.4f}"
            )
    print("=" * 72 + "\n")

    # --- Step 4: plot ---
    if show_plot:
        plot_flux_curvature(
            curvature_datasets,
            detector_positions_mm=detector_positions_mm,
            show_plot=show_plot,
            save_csv=save_csv,
            title_suffix=title_suffix,
        )

    return curvature_datasets


def compute_and_plot_T4_heatmap(
    times_to_store,
    *,
    mode="marshak_ablation",
    wall_material="Gold",
    use_seconds=True,
    vary_rho=True,
    lam_eff=None,
    power=None,
    show_plot=True,
    save_csv=True,
    time_snapshot_ns=2.0,
):
    """
    Compute and plot a 2D spatial heatmap of T^4(r, z) at a selected time snapshot (default 2.0 ns).
    The heatmap is plotted for r >= 0 and z >= 0.
    """
    # Determine defaults based on material
    if lam_eff is None:
        lam_eff = (Material in ["Ta2O5", "SiO2"])
    if power is None:
        power = 1 if (Material in ["Ta2O5", "SiO2"]) else 2

    # --- Step 1: run the solver ---
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
    bessel_data = result[5] if len(result) > 5 else {}
    times = np.asarray(times_to_store, dtype=float)

    if not bessel_data:
        print("Error: No bessel_data available in solver. Cannot compute T^4 heatmap.")
        return

    # Find snapshot closest to time_snapshot_ns (e.g. 2.0 ns)
    available = np.array(list(bessel_data.keys()), dtype=float)
    t_closest = float(available[np.argmin(np.abs(available - time_snapshot_ns))])
    data = bessel_data[t_closest]

    # Resolve exponents
    exponent = 1.0 / (4.0 + alpha - beta)
    if wall_material == "Gold":
        exponent_wall = 1.0 / (4.0 + alpha_gold - beta_gold)
    elif wall_material == "Copper":
        exponent_wall = 1.0 / (4.0 + alpha_copper - beta_copper)
    elif wall_material == "Be":
        exponent_wall = 1.0 / (4.0 + alpha_be - beta_be)
    elif wall_material == "Vacuum":
        exponent_wall = 0.0
    else:
        exponent_wall = exponent

    # Grids
    r_mesh_foam = np.asarray(data.get('r_grid', R_GRID_DEFAULT), dtype=float)
    r_mesh = np.asarray(data.get('r_gold_grid', r_mesh_foam), dtype=float)
    z_mesh = np.asarray(data.get('z_grid', z), dtype=float)
    R_mesh, Z_mesh = np.meshgrid(r_mesh, z_mesh)

    z_F_radial = np.asarray(data['z_F_radial'], dtype=float)
    Ts_t = np.interp(t_closest, times, Ts)

    # Compute foam temperature mesh
    T_mesh_foam = _compute_temperature_mesh(z_mesh, z_F_radial, Ts_t, exponent)

    # Map foam solution onto full radial grid
    T_mesh_plot = np.zeros((z_mesh.size, r_mesh.size), dtype=float)
    foam_domain = r_mesh <= R_cm
    for i_z in range(z_mesh.size):
        T_mesh_plot[i_z, foam_domain] = np.interp(
            r_mesh[foam_domain],
            r_mesh_foam,
            T_mesh_foam[i_z],
            left=0.0,
            right=0.0,
        )

    # Wall profile
    penetration_profile = data.get('wall_penetration_radius_profile')
    if penetration_profile is None:
        penetration_profile = np.full_like(z_mesh, R_cm, dtype=float)
    else:
        penetration_profile = np.asarray(penetration_profile, dtype=float)

    shock_profile = data.get('shock_penetration_radius_profile')
    if shock_profile is not None:
        shock_profile = np.asarray(shock_profile, dtype=float)

    foam_mask = data.get('ablation_foam_mask')
    wall_mask = data.get('ablation_wall_mask')
    if foam_mask is not None:
        foam_mask = np.asarray(foam_mask, dtype=bool)
    if wall_mask is not None:
        wall_mask = np.asarray(wall_mask, dtype=bool)

    ablation = "ablation" in mode
    if ablation:
        if (
            foam_mask is not None
            and wall_mask is not None
            and foam_mask.shape == T_mesh_foam.shape
            and wall_mask.shape == T_mesh_foam.shape
        ):
            T_wall_profile = _compute_wall_heyney_horizontal_profile(
                T_mesh_foam,
                foam_mask,
                wall_mask,
                r_mesh_foam,
                exponent_wall,
                is_ablation=True,
                r_mesh_wall=r_mesh,
                penetration_radius_profile=penetration_profile,
                shock_radius_profile=shock_profile,
            )
            wall_valid = np.isfinite(T_wall_profile)
            if np.any(wall_valid):
                T_mesh_plot[wall_valid] = T_wall_profile[wall_valid]
    else:
        if wall_material != "Vacuum":
            T_wall_profile = _compute_wall_heyney_horizontal_profile(
                T_mesh_foam,
                foam_mask=None,
                wall_mask=None,
                r_mesh=r_mesh_foam,
                exponent_wall=exponent_wall,
                is_ablation=False,
                r_mesh_wall=r_mesh,
                penetration_radius_profile=penetration_profile,
                shock_radius_profile=shock_profile,
            )
            wall_valid = np.isfinite(T_wall_profile)
            if np.any(wall_valid):
                T_mesh_plot[wall_valid] = T_wall_profile[wall_valid]

    # Convert shock boundary to NaN if needed
    show_shock = True
    if show_shock and shock_profile is not None:
        shock_mask = np.isfinite(shock_profile)
        if np.any(shock_mask):
            T_mesh_plot = np.array(T_mesh_plot, copy=True)
            for i_z, shock_r in enumerate(shock_profile):
                if np.isfinite(shock_r):
                    T_mesh_plot[i_z, r_mesh > shock_r] = np.nan

    # Raise temperature to the 4th power to get T^4
    T4_spatial = T_mesh_plot ** 4

    # Set unheated/unreached regions (where T4_spatial <= 1e-10) to NaN to show as white background
    T4_spatial = np.where(T4_spatial > 1e-10, T4_spatial, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_facecolor('white')

    pcm = ax.pcolormesh(
        R_mesh,
        Z_mesh,
        T4_spatial,
        shading='gouraud',
        cmap='Spectral_r',
        vmin=0.0,
    )

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046, shrink=0.5)
    cbar.set_label(r'$T^4$ [$\mathrm{heV}^4$]', fontsize=13, fontname='serif')

    # Draw horizontal dashed line at detector depth if z_pos_mm is provided

    ax.set_xlabel(r'$r$ [cm]', fontsize=14, fontname='serif')
    ax.set_ylabel(r'$z$ [cm]', fontsize=14, fontname='serif')

    ax.set_xlim(0.0, float(r_mesh[-1]))
    ax.set_ylim(0.0, L/2)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    plt.tight_layout()

    save_figure(f"T4_spatial_heatmap_{t_closest:.1f}ns.png", model1_5=False, model2_D=True, dpi=250, bbox_inches='tight')

    if show_plot:
        plt.show()
    plt.close()

    # Save to CSV if requested
    if save_csv:
        out_dir = BASE_DIR / "Data_new" / Experiment / Material / "2D_shape" / "T4_spatial_heatmap"
        out_dir.mkdir(parents=True, exist_ok=True)
        flat_r = R_mesh.flatten()
        flat_z = Z_mesh.flatten()
        flat_T4 = T4_spatial.flatten()
        df = pd.DataFrame({"r_cm": flat_r, "z_cm": flat_z, "T4_hev4": flat_T4})
        csv_path = out_dir / f"T4_spatial_heatmap_{t_closest:.1f}ns.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved T^4 spatial heatmap data -> {csv_path}")


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("STARTING STANDALONE RADIATION FLUX COMPUTATION")
    print("=" * 72)
    
    print(f"Active Material: {Material}")
    
    # Select default time array based on Material (consistent with comparison.py)
    if Material == "SiO2":
        times = np.linspace(0.01, 4.0, 1000)
        detector_positions_mm = [0.5, 1.0, 1.25]
    elif Material == "SiO2_low_energy":
        times = np.linspace(0.01, 15.0, 1000)
        detector_positions_mm = [1.0]
    elif Material == "C11H16Pb0.3852":
        times = np.linspace(0.01, 1.0, 1000)
    elif Material in ["C6H12", "C6H12Cu0.394"]:
        times = np.linspace(0.01, 2.0, 1000)
    elif Material == "Ta2O5":
        times = np.linspace(0.01, 4.0, 1000)
        detector_positions_mm = [0.25, 0.5, 0.75, 1.0]
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
        vary_rho = True,
        wall_material="Gold",
        detector_positions_mm=detector_positions_mm if 'detector_positions_mm' in locals() else [1.0],
        show_plot=True,
        save_csv=True,
    )
    print("Done! CSV results saved under Data_new/ and plot saved under Figures_new/.")

    # --- Flux curvature analysis (2D radial profiles) ---
    print("\n" + "=" * 72)
    print("STARTING FLUX CURVATURE (RADIAL PROFILE) COMPUTATION")
    print("=" * 72)

    # Select snapshot times based on material
    if Material in ["SiO2", "SiO2_Moore"]:
        curvature_snapshots = [1.0]
    elif Material == "SiO2_low_energy":
        curvature_snapshots = [9.5]
    elif Material == "C8H8":
        curvature_snapshots = [0.5, 1.0, 1.3]
    elif Material in ["C15H20O6", "C15H20O6Au0.172"]:
        curvature_snapshots = [1.0, 2.0, 2.5]
    elif Material == "Ta2O5":
        curvature_snapshots = [2]
    else:
        curvature_snapshots = [1.0, 2.0, 2.5]

    compute_and_plot_flux_curvature(
        times,
        times_ns_snapshots=curvature_snapshots,
        mode="marshak_wall_loss",
        wall_material="Gold",
        vary_rho=False,
        show_plot=True,
        save_csv=True,
        detector_positions_mm=detector_positions_mm if 'detector_positions_mm' in locals() else [1.0], 
        title_suffix=" (gold loss)",
    )
    compute_and_plot_flux_curvature(
        times,
        times_ns_snapshots=curvature_snapshots,
        mode="marshak_ablation",
        wall_material="Gold",
        vary_rho=True,
        lam_eff=True,
        power = 1.5,
        show_plot=True,
        save_csv=True,
        detector_positions_mm=detector_positions_mm if 'detector_positions_mm' in locals() else [1.0], 
        title_suffix=" (ablation)",
    )
    
    # --- New: Curvature post breakout comparison ---
    if Material == "Ta2O5":
        print("\n" + "=" * 72)
        print("STARTING FLUX CURVATURE POST-BREAKOUT COMPARISON")
        print("=" * 72)
        plot_flux_curvature_post_breakout(
            times,
            mode="marshak_ablation",
            wall_material="Gold",
            vary_rho=True,
            lam_eff=True,
            power=1,
            detector_positions_mm=[0.25, 0.5, 0.75, 1.0],
            delay_ns=0.5,
            show_plot=True,
        )
        # print("\n" + "=" * 72)
        # print("STARTING FLUX CURVATURE POST-BREAKOUT COMPARISON")
        # print("=" * 72)
        # plot_flux_curvature_post_breakout(
        #     times,
        #     mode="marshak_wall_loss",
        #     wall_material="Gold",
        #     vary_rho=False,
        #     lam_eff=True,
        #     power=1,
        #     detector_positions_mm=[0.25, 0.5, 0.75, 1.0],
        #     delay_ns=0.5,
        #     show_plot=True,
        # )
    elif Material == "SiO2":
        print("\n" + "=" * 72)
        print("STARTING FLUX CURVATURE POST-BREAKOUT COMPARISON")
        print("=" * 72)
        plot_flux_curvature_post_breakout(
            times,
            mode="marshak_ablation",
            wall_material="Gold",
            vary_rho=True,
            lam_eff=True,
            power=1,
            detector_positions_mm=[1.0],
            delay_ns=0.4 - 0.15,
            show_plot=True,
        )

    # --- New: T^4 Space-Time Heatmap at 1.0 mm ---
    print("\n" + "=" * 72)
    print("STARTING T^4 HEATMAP AT 1.0 mm")
    print("=" * 72)
    compute_and_plot_T4_heatmap(
        times,
        mode="marshak_ablation",
        wall_material="Gold",
        vary_rho=True,
        show_plot=True,
        save_csv=True,
        time_snapshot_ns = 6,
    )
        
    print("Done! Flux curvature results saved.")
