"""
Flux-curvature comparison: Be vs Gold coating at z = 0.5 mm, 200 ps after breakout.
Uses radiation_flux.py machinery (plot_flux_curvature_post_breakout internals)
to compute the model radial flux profile for both wall materials, normalised
to unity at r = 0, and overlays the experimental curves from Back et al.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set material BEFORE importing anything that reads parameters at import time
import parameters as _parameters
_parameters.Material = "Ta2O5"
_parameters.Experiment = "Back"

from model_main import analytic_wave_front_dispatch, BASE_DIR
from parameters import R_cm
from radiation_flux import (
    _numeric_time_keys,
    compute_flux_curvature_at_position,
    detect_arrival_time,
    SIGMA_SB,
)

# ── Plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'text.usetex': True,
    'axes.unicode_minus': False,
    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.linewidth': 2.0,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 14,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
})

# ── Configuration ─────────────────────────────────────────────────────────
DETECTOR_MM = 0.5            # detector position [mm]
DELAY_NS    = 0.2            # 200 ps after breakout
TIMES       = np.linspace(0.01, 4.0, 1000)

LAM_EFF   = False
POWER     = 1

WALL_CONFIGS = [
    {"wall_material": "Gold", "color": "#e6194b", "label": "Au (model)"},
    {"wall_material": "Be",   "color": "#4363d8", "label": "Be (model)"},
]

# ── Experimental data paths ───────────────────────────────────────────────
EXP_DATA_DIR = (
    BASE_DIR / "Data_new" / "Back" / "Ta2O5" / "article" / "flux_curvature"
)
EXP_FILES = {
    "Gold": EXP_DATA_DIR / "gold_curve.csv",
    "Be":   EXP_DATA_DIR / "be_curve.csv",
}

# ── Helper: run the solver and extract flux profile at breakout + delay ───
def _compute_profile(wall_material):
    """Return (r_mm_symmetric, flux_normalised_symmetric) for one wall material."""
    result = analytic_wave_front_dispatch(
        TIMES,
        use_seconds=True,
        mode="marshak_ablation" if wall_material == "Gold" else "marshak_wall_loss",
        wall_material=wall_material,
        vary_rho= True if wall_material == "Gold" else False,
        lam_eff=LAM_EFF,
        power=POWER,
    )
    plt.close('all')  # close solver's internal plots
    xF          = result[0]
    Ts          = result[1]
    bessel_data = result[5] if len(result) > 5 else {}
    times       = np.asarray(TIMES, dtype=float)

    if not bessel_data:
        raise RuntimeError(f"No bessel_data returned for wall_material={wall_material}")

    z_cm = DETECTOR_MM / 10.0

    # Breakout time
    _, t_breakout = detect_arrival_time(z_cm, times, xF)
    if t_breakout is None:
        raise RuntimeError(f"Front never reaches z = {DETECTOR_MM} mm for {wall_material}")

    t_target = t_breakout + DELAY_NS
    if DETECTOR_MM == 0.5:
        t_target += 0.1          # same adjustment as radiation_flux.py for 0.5 mm Ta2O5

    # Closest bessel snapshot
    available = _numeric_time_keys(bessel_data)
    t_closest = float(available[np.argmin(np.abs(available - t_target))])
    snapshot  = bessel_data[t_closest]

    r_grid_snap = np.asarray(snapshot['r_grid'], dtype=float)
    z_F_radial  = np.asarray(snapshot['z_F_radial'], dtype=float)
    Ts_t        = float(np.interp(t_closest, times, Ts))

    ablation = True if wall_material == "Gold" else False
    ds = compute_flux_curvature_at_position(
        z_cm, r_grid_snap, z_F_radial, Ts_t,
        snapshot=snapshot, wall_material=wall_material, ablation=ablation,
    )

    r_mm     = ds['r_grid'] * 10.0
    flux_raw = ds['flux_radial']

    # Normalise to 1 at r = 0
    center_val = flux_raw[0]
    if center_val > 0:
        flux_norm = flux_raw / center_val
    else:
        flux_norm = flux_raw

    # Symmetric profile
    r_sym    = np.concatenate((-r_mm[::-1], r_mm[1:]))
    flux_sym = np.concatenate((flux_norm[::-1], flux_norm[1:]))

    print(f"  {wall_material}: breakout = {t_breakout:.3f} ns, "
          f"snapshot = {t_closest:.3f} ns, center flux = {center_val:.4e}")

    return r_sym, flux_sym


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))

    for cfg in WALL_CONFIGS:
        wm = cfg["wall_material"]
        print(f"\nComputing model profile for {wm}...")
        r_sym, flux_sym = _compute_profile(wm)
        ax.plot(r_sym, flux_sym, color=cfg["color"], linewidth=2.2, label=cfg["label"])

        # Overlay experimental data
        exp_path = EXP_FILES.get(wm)
        if exp_path is not None and exp_path.exists():
            df_exp = pd.read_csv(exp_path)
            x_exp  = df_exp['x'].to_numpy()
            y_exp  = df_exp['y'].to_numpy()

            # Normalise to 1 at x closest to 0
            idx0  = np.argmin(np.abs(x_exp))
            y_ref = y_exp[idx0]
            if y_ref > 0:
                y_exp = y_exp / y_ref

            # Make symmetric
            x_sym = np.concatenate((-x_exp[::-1], x_exp[1:]))
            y_sym = np.concatenate((y_exp[::-1], y_exp[1:]))

            ax.plot(
                x_sym, y_sym,
                color=cfg["color"], linestyle='--', linewidth=2.0,
                label=cfg["label"].replace("model", "expt."),
            )

    ax.set_xlabel(r"$r$ [mm]")
    ax.set_ylabel(r"Flux [normalised]")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(loc='upper right', prop={'family': 'serif'})
    plt.tight_layout()

    # Save next to this script
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "flux_curvature_be_vs_gold.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    print(f"\nSaved figure -> {out_path}")

    # ── T⁴ Heatmaps: side-by-side Au vs Be ──────────────────────────────
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from shape_2D_analytical_model import _compute_wall_heyney_horizontal_profile, _compute_temperature_mesh
    from parameters import (
        alpha, beta, alpha_gold, beta_gold, alpha_be, beta_be, z, L,
    )

    heatmap_configs = [
        {"wall_material": "Gold", "mode": "marshak_ablation",  "title": "Au sleeve"},
        {"wall_material": "Be",   "mode": "marshak_wall_loss", "title": "Be sleeve"},
    ]

    fig_hm, axes_hm = plt.subplots(1, 2, figsize=(16, 7))

    # We need a common vmax across both panels
    vmax_global = 0.0
    heatmap_data = []

    for hcfg in heatmap_configs:
        wm   = hcfg["wall_material"]
        mode = hcfg["mode"]
        print(f"\nComputing T^4 heatmap for {wm}...")

        result = analytic_wave_front_dispatch(
            TIMES,
            use_seconds=True,
            mode=mode,
            wall_material=wm,
            vary_rho=True if wm == "Gold" else False,
            lam_eff=LAM_EFF,
            power=POWER,
        )
        plt.close('all')
        # Re-create heatmap figure after close
        if not plt.fignum_exists(fig_hm.number):
            fig_hm, axes_hm = plt.subplots(1, 2, figsize=(16, 7))

        xF          = result[0]
        Ts          = result[1]
        bessel_data = result[5] if len(result) > 5 else {}
        times       = np.asarray(TIMES, dtype=float)

        if not bessel_data:
            print(f"  No bessel_data for {wm}, skipping heatmap.")
            heatmap_data.append(None)
            continue

        # Use the same snapshot time as the curvature plot
        z_cm = DETECTOR_MM / 10.0
        _, t_breakout = detect_arrival_time(z_cm, times, xF)
        t_snap = t_breakout + DELAY_NS
        if DETECTOR_MM == 0.5:
            t_snap -= 0.1

        available = _numeric_time_keys(bessel_data)
        t_closest = float(available[np.argmin(np.abs(available - t_snap))])
        data = bessel_data[t_closest]

        # Exponents
        exponent_foam = 1.0 / (4.0 + alpha - beta)
        if wm == "Gold":
            exponent_wall = 1.0 / (4.0 + alpha_gold - beta_gold)
        elif wm == "Be":
            exponent_wall = 1.0 / (4.0 + alpha_be - beta_be)
        else:
            exponent_wall = exponent_foam

        # Grids
        r_mesh_foam = np.asarray(data.get('r_grid'), dtype=float)
        r_mesh      = np.asarray(data.get('r_gold_grid', r_mesh_foam), dtype=float)
        z_mesh      = np.asarray(data.get('z_grid', z), dtype=float)
        R_mesh, Z_mesh = np.meshgrid(r_mesh, z_mesh)

        z_F_radial = np.asarray(data['z_F_radial'], dtype=float)
        Ts_t       = float(np.interp(t_closest, times, Ts))

        # Foam temperature
        T_mesh_foam = _compute_temperature_mesh(z_mesh, z_F_radial, Ts_t, exponent_foam)

        # Map onto full radial grid
        T_mesh_plot = np.zeros((z_mesh.size, r_mesh.size), dtype=float)
        foam_domain = r_mesh <= R_cm
        for i_z in range(z_mesh.size):
            T_mesh_plot[i_z, foam_domain] = np.interp(
                r_mesh[foam_domain], r_mesh_foam, T_mesh_foam[i_z],
                left=0.0, right=0.0,
            )

        # Wall profile
        pen_profile   = data.get('wall_penetration_radius_profile')
        shock_profile = data.get('shock_penetration_radius_profile')
        foam_mask     = data.get('ablation_foam_mask')
        wall_mask     = data.get('ablation_wall_mask')

        ablation = "ablation" in mode
        if ablation and foam_mask is not None and wall_mask is not None:
            T_wall = _compute_wall_heyney_horizontal_profile(
                T_mesh_foam, foam_mask, wall_mask, r_mesh_foam, exponent_wall,
                is_ablation=True, r_mesh_wall=r_mesh,
                penetration_radius_profile=pen_profile,
                shock_radius_profile=shock_profile,
            )
            valid = np.isfinite(T_wall)
            if np.any(valid):
                T_mesh_plot[valid] = T_wall[valid]
        elif wm != "Vacuum":
            T_wall = _compute_wall_heyney_horizontal_profile(
                T_mesh_foam, None, None, r_mesh_foam, exponent_wall,
                is_ablation=False, r_mesh_wall=r_mesh,
                penetration_radius_profile=pen_profile,
                shock_radius_profile=shock_profile,
            )
            valid = np.isfinite(T_wall)
            if np.any(valid):
                T_mesh_plot[valid] = T_wall[valid]

        # Mask beyond shock
        if shock_profile is not None:
            for i_z, sr in enumerate(shock_profile):
                if np.isfinite(sr):
                    T_mesh_plot[i_z, r_mesh > sr] = np.nan

        T4 = T_mesh_plot ** 4
        T4 = np.where(T4 > 1e-10, T4, np.nan)

        local_max = float(np.nanmax(T4))
        if local_max > vmax_global:
            vmax_global = local_max

        heatmap_data.append({
            "R_mesh": R_mesh, "Z_mesh": Z_mesh, "T4": T4,
            "t_ns": t_closest, "title": hcfg["title"],
        })
        print(f"  {wm}: snapshot = {t_closest:.3f} ns, T4_max = {local_max:.4e}")

    # Plot both panels with common colour scale
    for idx, (ax_hm, hd) in enumerate(zip(axes_hm, heatmap_data)):
        if hd is None:
            ax_hm.set_visible(False)
            continue
        ax_hm.set_facecolor('white')
        pcm = ax_hm.pcolormesh(
            10 * hd["R_mesh"], 10 * hd["Z_mesh"], hd["T4"],
            shading='gouraud', cmap='Spectral_r', vmin=0.0, vmax=vmax_global,
        )
        # Detector depth line
        ax_hm.axhline(y=DETECTOR_MM, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_hm.set_xlabel(r"$r$ [mm]")
        if idx == 0:
            ax_hm.set_ylabel(r"$z$ [mm]")
        ax_hm.set_title(hd["title"])
        ax_hm.set_xlim(0.0, 10 * float(hd["R_mesh"].max()))
        ax_hm.set_ylim(0.0, 10 * L / 2)
        ax_hm.set_aspect('equal', adjustable='box')
        ax_hm.grid(False)

    # Shared colourbar
    fig_hm.subplots_adjust(right=0.88)
    cbar_ax = fig_hm.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig_hm.colorbar(pcm, cax=cbar_ax)
    cbar.set_label(r'$T^4$ [$\mathrm{heV}^4$]')

    hm_path = out_dir / "T4_heatmap_be_vs_gold.png"
    fig_hm.savefig(str(hm_path), dpi=200, bbox_inches='tight')
    print(f"\nSaved heatmap -> {hm_path}")
    plt.show()
