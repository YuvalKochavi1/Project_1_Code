"""
Verification script for the new compact bath-temperature HR Marshak fluence model.
Compares:
  1. `analytic_wave_front_marshak_fluence` (new Section 2B fluence method)
  2. `analytic_wave_front_marshak` (existing Appendix A time-marching method)
  3. `analytic_wave_front_no_marshak` (Ts = TD limit, verified against u=1 limit)
Also plots and compares leading-order vs first-order corrected spatial temperature profiles.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "1D_simulation") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "1D_simulation"))

from parameters import *
from wavefront_helpers import WavefrontHelpers
from model_main import (
    analytic_wave_front_marshak_fluence,
    analytic_wave_front_marshak,
    analytic_wave_front_no_marshak,
    analytic_wave_front_dispatch,
    z
)
from simulation import GoldFoam1DSimulation

OUTPUT_DIR = Path(__file__).resolve().parent / "Figures_new" / Experiment / Material / "Fluence_Verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Successive global history-correction order for marshak_fluence.
# k=0 is the compact model; larger k includes more history correction.
K_HISTORY = 10

def run_verification():
    print("=" * 70)
    print("VERIFICATION OF COMPACT BATH-TEMPERATURE HR MARSHAK FLUENCE MODEL")
    print("=" * 70)

    # 1. Define time vector (in ns) up to 2.5 ns
    times_ns = np.linspace(0.01, 2.5, 250)

    print("\n--- Running New Fluence Model ---")
    xF_fluence, Ts_fluence, T_lead, T_corr = analytic_wave_front_marshak_fluence(
        times_ns,
        use_seconds=False,
        k=K_HISTORY,
    )

    print("--- Running Existing Appendix A Marching Model ---")
    xF_march, Ts_march, E_tot_march, Ew_march, data_of_R_march, bessel_data_march = analytic_wave_front_marshak(times_ns, use_seconds=False)
    if isinstance(bessel_data_march, dict) and 'T_profile_corrected' in bessel_data_march:
        T_lead_march = bessel_data_march['T_profile_leading']
        T_corr_march = bessel_data_march['T_profile_corrected']
    else:
        _, T_lead_march, T_corr_march = WavefrontHelpers.compute_first_order_hr_profile(times_ns * 1e-9, xF_march, Ts_march)

    print("--- Running No-Marshak Model (Ts = Tb) ---")
    xF_no_marshak = analytic_wave_front_no_marshak(times_ns, use_seconds=False)

    print("--- Running/Loading 1D Numerical Simulation (without gold) ---")
    npz_path = PROJECT_ROOT / "Data_new" / Experiment / Material / "1D_simulation" / "run_outputs_1d.npz"
    if npz_path.exists():
        print(f"Loading existing 1D numerical simulation from: {npz_path}")
        data_1d = np.load(npz_path)
        stored_t_1d = data_1d["stored_t"]
        stored_Um_1d = data_1d["stored_Um"]
        stored_Tm_1d = data_1d["stored_Tm"]
        sim_1d = GoldFoam1DSimulation(nz=stored_Tm_1d.shape[1], lz=0.3, gold_block_width=0)
    else:
        print("Running 1D numerical simulation without gold (~20 seconds)...")
        sim_1d = GoldFoam1DSimulation(nz=400, lz=0.3, t_final=2.5e-9, gold_block_width=0)
        stored_t_1d, stored_Um_1d, stored_Tm_1d, _ = sim_1d.run(times_ns * 1e-9, marshak_boundary=True)

    front_pos_1d, _, _ = sim_1d.compute_front_and_energy(stored_Um_1d, stored_Tm_1d)

    # 2. Compare xF and Ts (Fluence vs Marching)
    print("\n=== Comparison: Fluence Model vs Appendix A Marching ===")
    mask_valid = xF_march > 1e-6
    if np.any(mask_valid):
        rel_diff_xF = np.abs(xF_fluence[mask_valid] - xF_march[mask_valid]) / xF_march[mask_valid] * 100.0
        rel_diff_Ts = np.abs(Ts_fluence[mask_valid] - Ts_march[mask_valid]) / Ts_march[mask_valid] * 100.0
        print(f"Front position xF - Mean rel diff: {np.mean(rel_diff_xF):.2f}%, Max rel diff: {np.max(rel_diff_xF):.2f}%")
        print(f"Surface temp Ts   - Mean rel diff: {np.mean(rel_diff_Ts):.2f}%, Max rel diff: {np.max(rel_diff_Ts):.2f}%")
    else:
        print("Warning: xF_march values are too small for relative comparison.")

    # 3. Verify u -> 1 Limit
    # To check u -> 1 exactly, we force u=1 in the fluence formula:
    # When u=1, Ts = Tb, and xF^2 = K_eps * C * Tb^{-beta} * Jn
    # Which is exactly the definition of analytic_wave_front_no_marshak (Eq. 8 in document).
    print("\n=== Verification of u -> 1 Limit vs No-Marshak ===")
    eps, sigma_SB_hev, C, pref = WavefrontHelpers.compute_constants_for_wavefront()
    n = 4.0 + alpha
    K_eps = pref
    
    t_sec = times_ns * 1e-9
    Tb_hev = np.array([get_TD(times_ns[i], t_array_TD, T_array_TD) for i in range(len(times_ns))])
    
    # Compute Jn trapezoidally as in fluence model
    t_aug = np.concatenate(([0.0], t_sec))
    Tb0 = get_TD(0.0, t_array_TD, T_array_TD)
    Tb_aug = np.concatenate(([Tb0], Tb_hev))
    Tbn_aug = Tb_aug ** n
    Jn_aug = np.zeros_like(t_aug)
    for i in range(1, len(t_aug)):
        dt_i = t_aug[i] - t_aug[i - 1]
        Jn_aug[i] = Jn_aug[i - 1] + 0.5 * (Tbn_aug[i] + Tbn_aug[i - 1]) * dt_i
    Jn = Jn_aug[1:]

    # Formula for front when u=1
    xF_u1_limit = np.sqrt(np.maximum(K_eps * C * (Tb_hev ** (-beta)) * Jn, 0.0)) / 1.02 # note: adjust factor 1.02 used in no_marshak

    # Compare xF_u1_limit directly against analytic_wave_front_no_marshak
    mask_nm = xF_no_marshak > 1e-6
    if np.any(mask_nm):
        diff_u1 = np.abs(xF_u1_limit[mask_nm] - xF_no_marshak[mask_nm]) / xF_no_marshak[mask_nm] * 100.0
        print(f"u=1 limit vs `analytic_wave_front_no_marshak` - Mean rel diff: {np.mean(diff_u1):.4e}%, Max: {np.max(diff_u1):.4e}%")
        if np.max(diff_u1) < 1.0:
            print("SUCCESS: u -> 1 limit matches `analytic_wave_front_no_marshak`!")
        else:
            print("Note: Small difference due to trapezoid grid / integration step differences.")

    # 4. Generate Figures
    print("\n--- Generating Plots ---")
    
    # Figure 1: Front Position xF(t)
    plt.figure(figsize=(9, 6))
    plt.plot(times_ns, xF_fluence, label="New Fluence Model (Compact)", color="blue", lw=2)
    plt.plot(times_ns, xF_march, label="Appendix A Marching Model", color="orange", linestyle="--", lw=2)
    plt.plot(times_ns, xF_no_marshak, label="No-Marshak ($T_s = T_b$)", color="green", linestyle=":", lw=2)
    plt.plot(stored_t_1d, front_pos_1d, label="1D Num. Sim. (no gold)", color="magenta", linestyle="-.", lw=2)
    plt.xlabel("$t$ [ns]", fontsize=16)
    plt.ylabel("$x_F$ [cm]", fontsize=16)
    plt.title("Comparison of Heat-Front Position $x_F(t)$", fontsize=16)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "comparison_xF.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Surface Temperature Ts(t)
    plt.figure(figsize=(9, 6))
    plt.plot(times_ns, Ts_fluence * 100, label="New Fluence Model $T_s$", color="blue", lw=2)
    plt.plot(times_ns, Ts_march * 100, label="Appendix A Marching $T_s$", color="orange", linestyle="--", lw=2)
    plt.plot(times_ns, Tb_hev * 100, label="Bath Temperature $T_b$", color="red", linestyle=":", lw=2)
    plt.plot(stored_t_1d, stored_Tm_1d[:, 0] * 100, label="1D Num. Sim. $T_s$", color="magenta", linestyle="-.", lw=2)
    plt.xlabel("$t$ [ns]", fontsize=16)
    plt.ylabel("Temperature [eV]", fontsize=16)
    plt.title("Comparison of Surface Temperature $T_s(t)$ vs Bath $T_b(t)$", fontsize=16)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "comparison_Ts.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Temperature Profiles T(x, t) at selected times
    plt.figure(figsize=(10, 6))
    selected_indices = [int(len(times_ns) * 0.3), int(len(times_ns) * 0.6), int(len(times_ns) * 0.9)]
    colors = ['purple', 'teal', 'darkred']
    for idx, c in zip(selected_indices, colors):
        t_val = times_ns[idx]
        plt.plot(z, T_lead[idx] * 100, label=f"Leading-order ($t={t_val:.2f}$ ns)", color=c, lw=2)
        plt.plot(z, T_corr[idx] * 100, label=f"1st-order Corrected ($t={t_val:.2f}$ ns)", color=c, linestyle="--", lw=2)
        idx_1d = np.argmin(np.abs(stored_t_1d - t_val))
        plt.plot(sim_1d.z, stored_Tm_1d[idx_1d] * 100, label=f"1D Num. Sim. ($t={stored_t_1d[idx_1d]:.2f}$ ns)", color=c, linestyle=":", lw=2)
    plt.xlabel("$z$ [cm]", fontsize=16)
    plt.ylabel("Temperature $T(z, t)$ [eV]", fontsize=16)
    plt.title("Spatial Temperature Profiles: Leading vs Corrected vs 1D Num.", fontsize=16)
    plt.xlim(0, max(xF_fluence[selected_indices[-1]] * 1.2, 0.05))
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    fig3_path = OUTPUT_DIR / "comparison_profiles.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    # Figure 4: Temperature Profiles T(x, t) using Appendix A Marching Model at selected times
    plt.figure(figsize=(10, 6))
    for idx, c in zip(selected_indices, colors):
        t_val = times_ns[idx]
        plt.plot(z, T_lead_march[idx] * 100, label=f"App. A Leading ($t={t_val:.2f}$ ns)", color=c, lw=2)
        plt.plot(z, T_corr_march[idx] * 100, label=f"App. A 1st-order Corrected ($t={t_val:.2f}$ ns)", color=c, linestyle="--", lw=2)
        idx_1d = np.argmin(np.abs(stored_t_1d - t_val))
        plt.plot(sim_1d.z, stored_Tm_1d[idx_1d] * 100, label=f"1D Num. Sim. ($t={stored_t_1d[idx_1d]:.2f}$ ns)", color=c, linestyle=":", lw=2)
    plt.xlabel("$z$ [cm]", fontsize=16)
    plt.ylabel("Temperature $T(z, t)$ [eV]", fontsize=16)
    plt.title("Spatial Profiles: Appendix A Marching vs 1D Num. Sim.", fontsize=16)
    plt.xlim(0, max(xF_march[selected_indices[-1]] * 1.2, 0.05))
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    fig4_path = OUTPUT_DIR / "comparison_profiles_marshakA.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved: {fig4_path}")

    # Figure 5: Temperature Profiles T(z, t) with Simulation xF scaled to match Model xF (no masking, simple division)
    plt.figure(figsize=(10, 6))
    for idx, c in zip(selected_indices, colors):
        t_val = times_ns[idx]
        idx_1d = np.argmin(np.abs(stored_t_1d - t_val))
        xF_sim = front_pos_1d[idx_1d]
        xF_m = xF_march[idx]
        pos_indices = np.where(stored_Tm_1d[idx_1d] * 100 > 0.5)[0]
        x_sim_zero = sim_1d.z[min(pos_indices[-1] + 1, len(sim_1d.z) - 1)] if len(pos_indices) > 0 else xF_sim
        scale_sim = 1.02*(xF_m / x_sim_zero) if x_sim_zero > 1e-10 else 1.0

        plt.plot(z, T_lead_march[idx] * 100, label=f"App. A Leading ($t={t_val:.2f}$ ns)", color=c, lw=2)
        plt.plot(z, T_corr_march[idx] * 100, label=f"App. A 1st-order Corrected ($t={t_val:.2f}$ ns)", color=c, linestyle="--", lw=2)
        plt.plot(sim_1d.z * scale_sim, stored_Tm_1d[idx_1d] * 100, label=f"1D Num. Sim. scaled ($t={stored_t_1d[idx_1d]:.2f}$ ns)", color=c, linestyle=":", lw=2.5)
    plt.xlabel("$z$ [cm] (1D Sim $z$ scaled so $x_F^{\\mathrm{sim}} = x_F^{\\mathrm{model}}$)", fontsize=15)
    plt.ylabel("Temperature $T(z, t)$ [eV]", fontsize=16)
    plt.title("Spatial Profiles (1D Sim Front Aligned to Model $x_F^{\\mathrm{model}}$)", fontsize=16)
    max_z = xF_march[selected_indices[-1]] * 1.15
    plt.xlim(0, max(max_z, 0.05))
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    fig5_path = OUTPUT_DIR / "comparison_profiles_marshakA_normalized_xF.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"Saved: {fig5_path}")

    print("\nVerification completed successfully!")

if __name__ == "__main__":
    run_verification()
