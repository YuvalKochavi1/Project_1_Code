"""
Radiation Flux Computation from 2D Simulation Data
====================================================

Computes the radiation flux Φ(t) = σ_SB · T(z,t)⁴  at detector positions
using the 2D simulation temperature field stored_Tm (shape: Nt × Nz × Nr).

This mirrors the analytical model's flux computation in radiation_flux.py,
but operates directly on the numerically solved temperature field rather
than the Henyey self-similar profile.

Two types of flux analysis:
  1. **Flux vs time** — Φ(t) at fixed axial depth z, averaged / sampled
     over the radial coordinate r  (analogous to compute_flux_at_position).
  2. **Flux curvature** — radial profile Φ(r) at a fixed depth z and a
     chosen time snapshot (analogous to compute_flux_curvature_at_position).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from parameters import kind_of_D_face
from simulation_2d_core import a_kelvin, c_cgs, K_per_Hev, a_hev



# --------------------------------------------------------------------------
# Physical constants (matching the model's radiation_flux.py)
# --------------------------------------------------------------------------
# In HeV units: σ_SB = a·c/4  where a = a_hev, c = 30 cm/ns
SIGMA_SB_HEV = a_hev * 30.0 / 4.0          # HeV unit system (cm, ns)
SIGMA_SB_CGS = a_kelvin * c_cgs / 4.0      # CGS unit system (cm, s)

DETECTOR_POSITIONS_MM = [0.25,0.5, 0.75, 1.0]     # default detector positions


# ==========================================================================
# 1. Flux vs Time  —  Φ(t) at a fixed axial depth
# ==========================================================================

def compute_flux_vs_time(
    sim,
    stored_t,
    stored_Tm,
    *,
    z_pos_cm,
    r_index=0,
):
    """
    Compute the radiation flux Φ(t) = σ_SB · T(z,t)⁴  at a fixed axial
    depth *z_pos_cm* from the simulation temperature field, for a single
    radial index.

    Parameters
    ----------
    sim : SelfSimilarDiffusion2D
        The simulation object (needed for grid info and unit system).
    stored_t : 1-D array, shape (Nt,)
        Stored time values [seconds in CGS, ns in hev|ns].
    stored_Tm : 3-D array, shape (Nt, Nz, Nr)
        Material temperature at each stored snapshot.
    z_pos_cm : float
        Axial detector position [cm].
    r_index : int
        Radial index to sample (default 0 = axis).

    Returns
    -------
    dict with keys:
        'times'          – time array (same as stored_t)
        'times_ns'       – time array in nanoseconds
        'z_pos_cm'       – detector position [cm]
        'z_pos_mm'       – detector position [mm]
        'r_index'        – radial index used
        'r_cm'           – radial position [cm]
        'T_at_detector'  – temperature at detector (Nt,)  [simulation units]
        'T_hev'          – temperature at detector [HeV]
        'flux_raw'       – un-normalised flux Φ(t)
        'i_arrival'      – index of first arrival
        't_arrival'      – arrival time [same units as stored_t]
    """
    stored_t  = np.asarray(stored_t, dtype=float)
    stored_Tm = np.asarray(stored_Tm, dtype=float)
    Nt = stored_Tm.shape[0]

    # Find the closest z-index on the simulation grid
    z_idx = int(np.argmin(np.abs(sim.z - z_pos_cm)))

    # Extract temperature time-series at (z_idx, r_index)
    T_series = stored_Tm[:, z_idx, r_index]   # (Nt,)

    # Convert to HeV for flux computation
    if sim.simulation_unit_system == "cgs":
        T_hev = T_series / K_per_Hev
        sigma_sb = SIGMA_SB_CGS
        times_ns = stored_t * 1e9
    else:
        T_hev = T_series
        sigma_sb = SIGMA_SB_HEV
        times_ns = stored_t

    # Stefan-Boltzmann flux
    flux_raw = sigma_sb * T_hev ** 4

    # Arrival detection: first time the temperature rises above cold bath
    if sim.simulation_unit_system == "cgs":
        T_cold = 300.0 / K_per_Hev  # HeV
    else:
        T_cold = 300.0 / K_per_Hev  # HeV
    threshold_factor = 1.5
    mask = T_hev > (threshold_factor * T_cold)
    if np.any(mask):
        i_arrival = int(np.argmax(mask))
        t_arrival = stored_t[i_arrival]
    else:
        i_arrival = None
        t_arrival = None

    return {
        'times':         stored_t,
        'times_ns':      times_ns,
        'z_pos_cm':      z_pos_cm,
        'z_pos_mm':      z_pos_cm * 10.0,
        'r_index':       r_index,
        'r_cm':          float(sim.r[r_index]),
        'T_at_detector': T_series,
        'T_hev':         T_hev,
        'flux_raw':      flux_raw,
        'i_arrival':     i_arrival,
        't_arrival':     t_arrival,
    }


def compute_flux_vs_time_radial_average(
    sim,
    stored_t,
    stored_Tm,
    *,
    z_pos_cm,
    r_max_cm=None,
):
    """
    Compute the radiation flux Φ(t) at a fixed depth *z_pos_cm*, averaged
    over the radial extent of the foam (r ∈ [0, r_max_cm]).

    This gives the total radiated power per unit area that a flat detector
    covering the full foam cross-section would measure.

    Parameters
    ----------
    sim : SelfSimilarDiffusion2D
        The simulation object.
    stored_t : 1-D array, shape (Nt,)
        Stored time values.
    stored_Tm : 3-D array, shape (Nt, Nz, Nr)
        Material temperature at each stored snapshot.
    z_pos_cm : float
        Axial detector position [cm].
    r_max_cm : float, optional
        Maximum radius for averaging.  Default: R_foam.

    Returns
    -------
    dict with keys:
        'times'          – time array (same as stored_t)
        'times_ns'       – time array in nanoseconds
        'z_pos_cm'       – detector position [cm]
        'z_pos_mm'       – detector position [mm]
        'T_avg_hev'      – radially-averaged temperature [HeV]
        'flux_raw'       – un-normalised flux Φ(t)
        'i_arrival'      – index of first arrival
        't_arrival'      – arrival time
    """
    stored_t  = np.asarray(stored_t, dtype=float)
    stored_Tm = np.asarray(stored_Tm, dtype=float)

    if r_max_cm is None:
        r_max_cm = sim.R_foam

    # Find closest z-index
    z_idx = int(np.argmin(np.abs(sim.z - z_pos_cm)))

    # Radial mask: cells within r_max_cm
    r_arr = np.asarray(sim.r, dtype=float)
    r_mask = r_arr <= r_max_cm
    r_sub = r_arr[r_mask]

    # Extract T(r) at this z for all times → (Nt, Nr_sub)
    T_slice = stored_Tm[:, z_idx, :][:, r_mask]

    # Convert to HeV
    if sim.simulation_unit_system == "cgs":
        T_hev_slice = T_slice / K_per_Hev
        sigma_sb = SIGMA_SB_CGS
        times_ns = stored_t * 1e9
    else:
        T_hev_slice = T_slice
        sigma_sb = SIGMA_SB_HEV
        times_ns = stored_t

    # Compute flux at every (t, r) point
    flux_2d = sigma_sb * T_hev_slice ** 4   # (Nt, Nr_sub)

    # Area-weighted radial average:  ∫ Φ(r) · 2πr dr  /  ∫ 2πr dr
    weights = 2.0 * np.pi * r_sub   # (Nr_sub,)
    total_weight = np.trapz(weights, r_sub)
    flux_avg = np.array([
        np.trapz(flux_2d[k, :] * weights, r_sub) / total_weight
        for k in range(flux_2d.shape[0])
    ])

    # Similarly average T for reference
    T_avg_hev = np.array([
        np.trapz(T_hev_slice[k, :] * weights, r_sub) / total_weight
        for k in range(T_hev_slice.shape[0])
    ])

    # Arrival detection
    if sim.simulation_unit_system == "cgs":
        T_cold = 300.0 / K_per_Hev
    else:
        T_cold = 300.0 / K_per_Hev
    threshold_factor = 1.5
    mask = T_avg_hev > (threshold_factor * T_cold)
    if np.any(mask):
        i_arrival = int(np.argmax(mask))
        t_arrival = stored_t[i_arrival]
    else:
        i_arrival = None
        t_arrival = None

    return {
        'times':       stored_t,
        'times_ns':    times_ns,
        'z_pos_cm':    z_pos_cm,
        'z_pos_mm':    z_pos_cm * 10.0,
        'T_avg_hev':   T_avg_hev,
        'flux_raw':    flux_avg,
        'i_arrival':   i_arrival,
        't_arrival':   t_arrival,
    }


def compute_flux_datasets(
    sim,
    stored_t,
    stored_Tm,
    *,
    detector_positions_mm=None,
    r_index=0,
    radial_average=False,
):
    """
    Compute and normalise radiation flux for multiple detector positions.

    Parameters
    ----------
    sim : SelfSimilarDiffusion2D
    stored_t : 1-D array (Nt,)
    stored_Tm : 3-D array (Nt, Nz, Nr)
    detector_positions_mm : list of float, optional
        Detector locations [mm].  Default: [0.5, 1.0, 1.5].
    r_index : int
        Radial index (used only when radial_average=False).
    radial_average : bool
        If True, average the flux over the foam radius instead of sampling
        at a single r_index.

    Returns
    -------
    datasets : list of dict
        One dict per detector.  Each dict contains the keys from
        compute_flux_vs_time plus 'flux_normalised'.
    """
    if detector_positions_mm is None:
        detector_positions_mm = DETECTOR_POSITIONS_MM

    datasets = []
    for z_mm in detector_positions_mm:
        z_cm = z_mm / 10.0
        if radial_average:
            ds = compute_flux_vs_time_radial_average(
                sim, stored_t, stored_Tm, z_pos_cm=z_cm,
            )
        else:
            ds = compute_flux_vs_time(
                sim, stored_t, stored_Tm, z_pos_cm=z_cm, r_index=r_index,
            )
        datasets.append(ds)

    # Global-peak normalisation (same convention as the model)
    global_peak = 0.0
    if datasets:
        global_peak = max(np.max(ds['flux_raw']) for ds in datasets)

    for ds in datasets:
        if global_peak <= 0.0:
            ds['flux_normalised'] = np.zeros_like(ds['flux_raw'])
        else:
            ds['flux_normalised'] = ds['flux_raw'] / global_peak

    return datasets


# ==========================================================================
# 2. Flux Curvature  —  radial profile Φ(r) at a fixed depth and time
# ==========================================================================

def compute_flux_curvature_at_snapshot(
    sim,
    stored_t,
    stored_Tm,
    *,
    z_pos_cm,
    t_snapshot,
):
    """
    Compute the radial flux profile Φ(r) = σ_SB · T(r)⁴ at a fixed axial
    depth *z_pos_cm* and a time closest to *t_snapshot*.

    Parameters
    ----------
    sim : SelfSimilarDiffusion2D
    stored_t : 1-D array (Nt,)
    stored_Tm : 3-D array (Nt, Nz, Nr)
    z_pos_cm : float
        Axial detector depth [cm].
    t_snapshot : float
        Desired time [same units as stored_t].

    Returns
    -------
    dict with keys:
        'r_grid'           – radial grid [cm]
        'r_grid_mm'        – radial grid [mm]
        'z_pos_cm'         – detector depth [cm]
        'z_pos_mm'         – detector depth [mm]
        't_actual'         – actual snapshot time used
        't_actual_ns'      – actual snapshot time in ns
        'T_radial_hev'     – temperature profile T(r) [HeV]
        'flux_radial'      – radiation flux Φ(r)
    """
    stored_t  = np.asarray(stored_t, dtype=float)
    stored_Tm = np.asarray(stored_Tm, dtype=float)

    # Find closest time index
    t_idx = int(np.argmin(np.abs(stored_t - t_snapshot)))
    t_actual = stored_t[t_idx]

    # Find closest z-index
    z_idx = int(np.argmin(np.abs(sim.z - z_pos_cm)))

    # Extract T(r) at this (z, t)
    T_radial = stored_Tm[t_idx, z_idx, :]   # (Nr,)

    # Convert to HeV
    if sim.simulation_unit_system == "cgs":
        T_hev = T_radial / K_per_Hev
        sigma_sb = SIGMA_SB_CGS
        t_actual_ns = t_actual * 1e9
    else:
        T_hev = T_radial.copy()
        sigma_sb = SIGMA_SB_HEV
        t_actual_ns = t_actual

    flux_radial = sigma_sb * T_hev ** 4

    r_arr = np.asarray(sim.r, dtype=float)

    return {
        'r_grid':        r_arr,
        'r_grid_mm':     r_arr * 10.0,
        'z_pos_cm':      z_pos_cm,
        'z_pos_mm':      z_pos_cm * 10.0,
        't_actual':      t_actual,
        't_actual_ns':   t_actual_ns,
        'T_radial_hev':  T_hev,
        'flux_radial':   flux_radial,
    }


def compute_flux_curvature_datasets(
    sim,
    stored_t,
    stored_Tm,
    *,
    t_snapshots,
    detector_positions_mm=None,
):
    """
    Build flux curvature datasets for multiple time snapshots and
    detector positions.

    Parameters
    ----------
    sim : SelfSimilarDiffusion2D
    stored_t : 1-D array (Nt,)
    stored_Tm : 3-D array (Nt, Nz, Nr)
    t_snapshots : list of float
        Time snapshots [same units as stored_t].
    detector_positions_mm : list of float, optional
        Detector depths [mm].  Default: [0.5, 1.0, 1.5].

    Returns
    -------
    results : dict
        results[t_ns][z_mm]  = dataset dict from compute_flux_curvature_at_snapshot
        results['global_peak'] = float (global max flux across all snapshots)
    """
    if detector_positions_mm is None:
        detector_positions_mm = DETECTOR_POSITIONS_MM

    results = {}
    all_peaks = []

    for t_snap in t_snapshots:
        t_snap = float(t_snap)
        results[t_snap] = {}

        for z_mm in detector_positions_mm:
            z_cm = z_mm / 10.0
            ds = compute_flux_curvature_at_snapshot(
                sim, stored_t, stored_Tm,
                z_pos_cm=z_cm, t_snapshot=t_snap,
            )
            results[t_snap][z_mm] = ds

            peak = float(np.max(ds['flux_radial']))
            if peak > 0:
                all_peaks.append(peak)

    # Normalisation
    global_peak = max(all_peaks) if all_peaks else 1.0
    results['global_peak'] = global_peak

    for t_key in list(results.keys()):
        if t_key == 'global_peak':
            continue
        for z_mm_key in results[t_key]:
            ds = results[t_key][z_mm_key]
            peak = float(np.max(ds['flux_radial']))
            if peak > 0:
                ds['flux_normalised'] = ds['flux_radial'] / peak
            else:
                ds['flux_normalised'] = np.zeros_like(ds['flux_radial'])

    return results


def compute_and_plot_flux_vs_time(
    sim,
    stored_t,
    stored_Tm,
    *,
    detector_positions_mm=None,
    r_index=0,
    show_plot=True,
    out_dir=None,
    title_suffix="",
):
    """
    End-to-end convenience function for flux vs time analysis in simulation:
      1. Extract temperature at each detector depth and time.
      2. Compute radiation flux Φ(t) = σ_SB · T⁴.
      3. Normalise, plot, and save.
    """
    if detector_positions_mm is None:
        detector_positions_mm = [0.5, 1.0, 1.5]
    datasets = compute_flux_datasets(
        sim,
        stored_t,
        stored_Tm,
        detector_positions_mm=detector_positions_mm,
        r_index=r_index,
    )
    print_flux_summary(datasets)
    plot_flux_vs_time(
        datasets,
        show_plot=show_plot,
        out_dir=out_dir,
        title_suffix=title_suffix,
    )
    return datasets



def print_flux_summary(datasets):
    """Print a summary table of arrival times and peak fluxes."""
    print("\n" + "=" * 72)
    print(f"{'z [mm]':>8}  {'t_arrival':>14}  {'Peak Φ':>16}  {'Peak T [HeV]':>14}")
    print("-" * 72)
    for ds in datasets:
        t_arr = ds.get('t_arrival')
        t_str = f"{t_arr:.4g}" if t_arr is not None else "N/A"
        peak_flux = np.max(ds['flux_raw'])
        peak_T = np.max(ds.get('T_hev', ds.get('T_avg_hev', [0])))
        print(f"{ds['z_pos_mm']:8.1f}  {t_str:>14}  {peak_flux:16.4e}  {peak_T:14.4f}")
    print("=" * 72 + "\n")


def print_flux_curvature_summary(curvature_datasets):
    """Print a summary table for flux curvature datasets."""
    print("\n" + "=" * 72)
    print("SIMULATION FLUX CURVATURE SUMMARY")
    print("-" * 72)
    print(f"{'t':>10}  {'z [mm]':>8}  {'Peak Φ':>16}  {'Center T [HeV]':>14}")
    print("-" * 72)
    for t_key in sorted(k for k in curvature_datasets.keys() if k != 'global_peak'):
        for z_mm in sorted(curvature_datasets[t_key].keys()):
            ds = curvature_datasets[t_key][z_mm]
            peak_flux = float(np.max(ds['flux_radial']))
            center_T = float(ds['T_radial_hev'][0])
            t_ns = ds.get('t_actual_ns', t_key)
            print(f"{t_ns:10.2f}  {z_mm:8.1f}  {peak_flux:16.4e}  {center_T:14.4f}")
    print("=" * 72 + "\n")


# ==========================================================================
# 4. Plotting helpers
# ==========================================================================

def plot_flux_vs_time(
    datasets,
    *,
    out_dir=None,
    filename=None,
    title_suffix="",
    show_plot=True,
):
    """
    Plot flux vs time for multiple detector positions (3-panel figure:
    raw flux, normalised flux, temperature).

    Parameters
    ----------
    datasets : list of dict
        As returned by compute_flux_datasets.
    out_dir : str or Path, optional
        Directory to save the figure.  If None, figure is not saved.
    filename : str, optional
        Filename for the saved figure.  Auto-generated if None.
    title_suffix : str
        Extra text for the figure title.
    show_plot : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4']

    # Panel 1: raw flux
    ax = axes[0]
    for ds, c in zip(datasets, colors):
        label = f"z = {ds['z_pos_mm']:g} mm"
        ax.plot(ds['times_ns'], ds['flux_raw'], color=c, linewidth=1.5, label=label)
        if ds['t_arrival'] is not None:
            t_ns = ds['t_arrival'] * 1e9 if 'times_ns' in ds and ds['times_ns'][0] != ds['times'][0] else ds['t_arrival']
            ax.axvline(t_ns, color=c, linestyle=':', alpha=0.5)
    ax.set_xlabel("Time (ns)", fontsize=14, fontname='serif')
    ax.set_ylabel(r"$\Phi$", fontsize=14, fontname='serif')
    ax.set_title("Raw Radiation Flux (Simulation)", fontsize=15, fontname='serif')
    ax.legend(prop={'family': 'serif'})
    ax.grid(True, alpha=0.3)

    # Panel 2: normalised flux
    ax = axes[1]
    for ds, c in zip(datasets, colors):
        label = f"z = {ds['z_pos_mm']:g} mm"
        ax.plot(ds['times_ns'], ds['flux_normalised'], color=c, linewidth=1.5, label=label)
        if ds['t_arrival'] is not None:
            t_ns = ds['t_arrival'] * 1e9 if ds['times_ns'][0] != ds['times'][0] else ds['t_arrival']
            ax.axvline(t_ns, color=c, linestyle=':', alpha=0.5)
    ax.set_xlabel("Time (ns)", fontsize=14, fontname='serif')
    ax.set_ylabel(r"$\Phi / \Phi_{\max}$", fontsize=14, fontname='serif')
    ax.set_title("Normalised Radiation Flux (Simulation)", fontsize=15, fontname='serif')
    ax.legend(prop={'family': 'serif'})
    ax.grid(True, alpha=0.3)

    # Panel 3: temperature
    ax = axes[2]
    T_key = 'T_hev' if 'T_hev' in datasets[0] else 'T_avg_hev'
    for ds, c in zip(datasets, colors):
        label = f"z = {ds['z_pos_mm']:g} mm"
        ax.plot(ds['times_ns'], ds[T_key], color=c, linewidth=1.5, label=label)
        if ds['t_arrival'] is not None:
            t_ns = ds['t_arrival'] * 1e9 if ds['times_ns'][0] != ds['times'][0] else ds['t_arrival']
            ax.axvline(t_ns, color=c, linestyle=':', alpha=0.5)
    ax.set_xlabel("Time (ns)", fontsize=14, fontname='serif')
    ax.set_ylabel("T (HeV)", fontsize=14, fontname='serif')
    ax.set_title("Temperature at Detectors (Simulation)", fontsize=15, fontname='serif')
    ax.legend(prop={'family': 'serif'})
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"2D Simulation — Radiation Flux Analysis{title_suffix}",
        fontsize=16, fontname='serif', y=1.02,
    )
    plt.tight_layout()

    # Save to disk
    if out_dir is not None:
        from csv_helpers import ensure_dir
        out_dir = Path(out_dir)
        ensure_dir(out_dir)
        if filename is None:
            safe_suffix = title_suffix.replace(' ', '_').replace('—', '-')
            filename = f"flux_vs_time{safe_suffix}.png"
        save_path = out_dir / filename
        fig.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"Saved figure -> {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_flux_curvature(
    curvature_datasets,
    *,
    detector_positions_mm=None,
    R_foam_cm=None,
    out_dir=None,
    filename=None,
    show_plot=True,
    title_suffix="",
):
    """
    Plot radial flux curvature profiles for each time snapshot and
    detector position.

    Parameters
    ----------
    curvature_datasets : dict
        As returned by compute_flux_curvature_datasets.
    detector_positions_mm : list of float, optional
    R_foam_cm : float, optional
        Foam radius [cm] for drawing the interface line.
    out_dir : str or Path, optional
        Directory to save the figure.  If None, figure is not saved.
    filename : str, optional
        Filename for the saved figure.  Auto-generated if None.
    show_plot : bool
    title_suffix : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if detector_positions_mm is None:
        detector_positions_mm = DETECTOR_POSITIONS_MM

    time_keys = sorted(k for k in curvature_datasets.keys() if k != 'global_peak')
    n_times = len(time_keys)
    n_detectors = len(detector_positions_mm)

    if n_times == 0 or n_detectors == 0:
        print("Warning: no data to plot for flux curvature.")
        return None

    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4']

    fig, axes = plt.subplots(
        n_detectors, n_times,
        figsize=(6 * n_times, 4.5 * n_detectors),
        squeeze=False,
    )

    for j, t_key in enumerate(time_keys):
        for i, z_mm in enumerate(detector_positions_mm):
            ax = axes[i][j]
            if z_mm not in curvature_datasets[t_key]:
                ax.set_visible(False)
                continue

            ds = curvature_datasets[t_key][z_mm]
            r_mm = ds['r_grid_mm']
            color = colors[i % len(colors)]

            # Symmetric profile
            r_sym = np.concatenate((-r_mm[::-1], r_mm[1:]))
            flux_sym = np.concatenate((ds['flux_normalised'][::-1], ds['flux_normalised'][1:]))
            ax.plot(r_sym, flux_sym, color=color, linewidth=2.0,
                    label=f"z = {z_mm:.1f} mm")

            # Foam-wall interface
            if R_foam_cm is not None:
                R_mm = R_foam_cm * 10.0
                ax.axvline(x=R_mm, color='gray', linestyle='--', alpha=0.7,
                           label='Foam-Wall interface')
                ax.axvline(x=-R_mm, color='gray', linestyle='--', alpha=0.7)

            t_ns = ds.get('t_actual_ns', t_key)
            ax.set_title(f"t = {t_ns:.2f} ns, z = {z_mm:.1f} mm",
                         fontsize=12, fontname='serif')
            ax.set_xlabel("r (mm)", fontsize=12, fontname='serif')
            if j == 0:
                ax.set_ylabel("Normalised flux", fontsize=12, fontname='serif')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(prop={'family': 'serif'}, fontsize=10)

    fig.suptitle(
        f"2D Simulation — Flux Curvature{title_suffix}",
        fontsize=14, fontname='serif', y=1.02,
    )
    plt.tight_layout()

    # Save to disk
    if out_dir is not None:
        from csv_helpers import ensure_dir
        out_dir = Path(out_dir)
        ensure_dir(out_dir)
        if filename is None:
            safe_suffix = title_suffix.replace(' ', '_').replace('—', '-')
            filename = f"flux_curvature{safe_suffix}.png"
        save_path = out_dir / filename
        fig.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"Saved figure -> {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_flux_curvature_post_breakout(
    sim,
    stored_t,
    stored_Tm,
    *,
    detector_positions_mm=None,
    delay_ns=0.5,
    show_plot=True,
    out_dir=None,
    mode="Simulation",
    material=None,
):
    """
    For the active material, plot the radial flux curvature at a fixed delay (e.g. 0.5 ns)
    after the breakout time for each detector position, all on the same graph, using global
    normalization to show relative attenuation.
    """
    if material is None:
        try:
            from simulation_2d_pipeline import Material
            material = Material
        except ImportError:
            material = "Ta2O5"

    if detector_positions_mm is None:
        if material == "Ta2O5":
            detector_positions_mm = [0.25, 0.5, 0.75, 1.0]
        else:
            detector_positions_mm = [0.5, 1.0, 1.5]

    import matplotlib.pyplot as plt
    import pandas as pd

    # Compute front positions along z at r=0
    z_front_array = sim.compute_front_at_r(stored_Tm, r_index=0, front_method="maxgrad")

    stored_t = np.asarray(stored_t, dtype=float)
    if sim.simulation_unit_system == "cgs":
        times_ns = stored_t * 1e9
        sigma_sb = SIGMA_SB_CGS
    else:
        times_ns = stored_t
        sigma_sb = SIGMA_SB_HEV

    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']

    # Pre-collect computed profiles so we can find global peak
    raw_profiles = []
    for idx, z_mm in enumerate(detector_positions_mm):
        z_cm = z_mm / 10.0
        
        # Detect arrival/breakout time
        mask = z_front_array >= z_cm
        if not np.any(mask):
            print(f"Heat front never reaches z = {z_mm:.2f} mm. Skipping.")
            continue
        i_breakout = int(np.argmax(mask))
        t_breakout_ns = times_ns[i_breakout]
            
        t_target_ns = t_breakout_ns + delay_ns
        
        # Find closest snapshot
        t_idx = int(np.argmin(np.abs(times_ns - t_target_ns)))
        t_closest_ns = times_ns[t_idx]
        
        z_idx = int(np.argmin(np.abs(sim.z - z_cm)))
        T_radial = stored_Tm[t_idx, z_idx, :]
        
        if sim.simulation_unit_system == "cgs":
            T_radial_hev = T_radial / K_per_Hev
        else:
            T_radial_hev = T_radial
            
        flux_radial = sigma_sb * (T_radial_hev ** 4)
        
        raw_profiles.append((z_mm, t_breakout_ns, t_closest_ns, flux_radial, T_radial_hev))

    if not raw_profiles:
        print("Error: No profiles could be computed.")
        return

    # Find the global maximum peak across all calculated profiles
    global_peak = max(float(np.max(p[3])) for p in raw_profiles)
    if global_peak <= 0:
        global_peak = 1.0

    plt.figure(figsize=(8, 6))

    # --- Pre-load experimental Data to calculate global normalisation factor (Ta2O5)
    exp_ta2o5 = {}
    y_norm = 1.0
    if material == "Ta2O5":
        base_ta_path = PROJECT_ROOT / "Data_new" / "Back" / "Ta2O5" / "article" / "flux_curvature"
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

    r_mm_last = None
    for idx, (z_mm, t_breakout, t_closest, flux_raw, T_radial_hev) in enumerate(raw_profiles):
        r_mm = sim.r * 10.0
        r_mm_last = r_mm
        
        flux_norm = flux_raw / global_peak
            
        r_sym = np.concatenate((-r_mm[::-1], r_mm[1:]))
        flux_sym = np.concatenate((flux_norm[::-1], flux_norm[1:]))
        
        label = f"z = {z_mm:.2f} mm ($t_{{br}} = {t_breakout:.2f}$ ns, plot $t = {t_closest:.2f}$ ns)"
        plt.plot(r_sym, flux_sym, color=colors[idx % len(colors)], linewidth=2.2, label=label)

        # Plot Ta2O5 experimental data
        if material == "Ta2O5":
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
                        zorder=5
                    )
                    break

    # Draw original Foam-Wall interface symmetrically
    plt.axvline(x=sim.R_foam * 10.0, color='gray', linestyle='--', alpha=0.7, label='Foam-Wall interface')
    plt.axvline(x=-sim.R_foam * 10.0, color='gray', linestyle='--')

    plt.xlabel("r (mm)", fontsize=13, fontname='serif')
    plt.ylabel("Normalized flux", fontsize=13, fontname='serif')
    plt.title(f"Flux Curvature at {delay_ns} ns after breakout — {material} ({mode})", fontsize=14, fontname='serif', y=1.02)
    if r_mm_last is not None:
        plt.xlim(-r_mm_last[-1], r_mm_last[-1])
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend(prop={'family': 'serif'}, fontsize=10)
    plt.tight_layout()

    if out_dir is not None:
        from csv_helpers import ensure_dir
        out_dir = Path(out_dir)
        ensure_dir(out_dir)
        save_path = out_dir / f"flux_curvature_post_breakout_{delay_ns}ns.png"
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"Saved figure -> {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def compute_and_plot_flux_curvature(
    sim,
    stored_t,
    stored_Tm,
    *,
    times_ns_snapshots=None,
    detector_positions_mm=None,
    show_plot=True,
    out_dir=None,
    title_suffix="",
):
    """
    End-to-end convenience function for flux curvature analysis in simulation:
      1. Extract curved front and radial temperature/flux profile for snapshots.
      2. At each detector depth, compute the radial temperature cross-section T(r, z_det, t)
         and the flux Φ(r) = σ_SB·T⁴.
      3. Normalise, plot, and save.
    """
    if times_ns_snapshots is None:
        times_ns_snapshots = [1.0, 2.0, 2.5]

    if sim.simulation_unit_system == "cgs":
        t_snapshots = [float(t) * 1e-9 for t in times_ns_snapshots]
    else:
        t_snapshots = [float(t) for t in times_ns_snapshots]

    # --- Step 1: flux curvature computation ---
    curvature_datasets = compute_flux_curvature_datasets(
        sim,
        stored_t,
        stored_Tm,
        t_snapshots=t_snapshots,
        detector_positions_mm=detector_positions_mm,
    )

    # --- Step 2: Print summary ---
    print_flux_curvature_summary(curvature_datasets)

    # --- Step 3: plot ---
    plot_flux_curvature(
        curvature_datasets,
        detector_positions_mm=detector_positions_mm,
        R_foam_cm=sim.R_foam,
        show_plot=show_plot,
        out_dir=out_dir,
        title_suffix=title_suffix,
    )

    return curvature_datasets
