"""
Analytical Models for Self-Similar Radiation Diffusion Wave Fronts
=====================================================================

This module provides analytical solutions and supporting calculations for:
  - Radiation diffusion with Marshak boundary conditions
  - Heat front position and evolution
  - Wall energy loss (ablation)
  - Effective density calculations
  - Albedo (reflectivity) from wall absorption

Key References:
  - Appendix A: Marshak boundary iteration algorithm
  - Equations (9)-(14): Front position, ablation, and wall loss models
"""

from parameters import *
from pathlib import Path
from eigen_bessel_solver import kappa_roots
from scipy import special
from wavefront_helpers import WavefrontHelpers
from wall_loss_model import WallLossModel
from ablation_model import AblationModel
from albedo_model import AlbedoModel
from analytical_wavefront_solver import AnalyticalWavefrontSolver

BASE_DIR = Path(__file__).resolve().parent
U_TILDA_DIR = BASE_DIR / "Data_new" / "u_tilda"

# ============================================================================
# SECTION 1: ANALYTICAL WAVEFRONT SOLUTIONS
# ============================================================================

# --- Mode 1: No Marshak (Ts = TD, direct solution) ---

def analytic_wave_front_no_marshak(times_to_store, *, use_seconds=True, lam_eff=False, power=2):
    """
    Mode 1 (your marshak_boundary=False):
      Ts(t) = TD(t)
      xF^2 = pref * C * H(t)^(-eps) * ∫_0^t H dt
      H(t) = Ts(t)^(4+alpha)
    Returns:
      xF (cm) in the same order as times_to_store
    """
    t_sec, order, t_sec_in = WavefrontHelpers.prepare_times(times_to_store, use_seconds=use_seconds)
    if t_sec.size == 0:
        return np.array([])

    eps, sigma_SB_hev, C, pref = WavefrontHelpers.compute_constants_for_wavefront()
    C0 = C
    K = (16.0 / (4.0 + alpha)) * (g * sigma_SB_hev) / (3.0 * f)
    p = mu - lambda_param - 2.0
    idx = np.argsort(t_sec)
    t_sec = t_sec[idx]
    # optionally drop duplicates
    mask = np.concatenate(([True], np.diff(t_sec) > 0))
    t_sec = t_sec[mask]
    g_eff_array = np.full_like(t_sec, g, dtype=float)
    C_eff_array = np.full_like(t_sec, C0, dtype=float)
    Iw = np.zeros_like(t_sec, dtype=float)  # integral of (g_eff/g)*rho^p over time
    if lam_eff:
        for i in range(1, len(t_sec)):
            dt_i = t_sec[i] - t_sec[i - 1]
            TD_i = np.interp(t_sec[i]*1e9, t_array_TD, T_array_TD) * 0.01
            lambda_ross = g * (TD_i**alpha) * (rho**(-lambda_param-1))
            lambda_geom = 2 * R_cm 
            lambda_param_eff = ((lambda_geom**(-power) + lambda_ross**(-power)))**(-1/power)
            g_eff = g * (lambda_param_eff / lambda_ross)
            g_eff_array[i] = g_eff
            # integrate weight * rho^p
            w_i   = g_eff_array[i]   / g
            w_im1 = g_eff_array[i-1] / g

            Iw[i] = Iw[i-1] + 0.5 * dt_i * (w_i + w_im1) * (rho**p)

            # C_eff from integral (NOT relaxation)
            C_eff_array[i] = (K / t_sec[i]) * Iw[i] if t_sec[i] > 0 else C0
    if t_sec[-1] > 1e-5:
        t_sec = t_sec * 1e-9
    t_ns = t_sec * 1e9
    Ts_hev = np.interp(t_ns, t_array_TD, T_array_TD) * 0.01
    H = Ts_hev ** (4.0 + alpha)

    # cumulative trapezoid I(t)=∫_0^t H dt
    if t_sec[0] > 0.0:
        t_aug = np.concatenate(([0.0], t_sec))
        H0 = (np.interp(0.0, t_array_TD, T_array_TD) * 0.01) ** (4.0 + alpha)
        H_aug = np.concatenate(([H0], H))

        if lam_eff:
            C_eff_aug = np.concatenate(([C_eff_array[0]], C_eff_array))

    else:
        t_aug = t_sec
        H_aug = H
        if lam_eff:
            C_eff_aug = C_eff_array

    I_aug = np.zeros_like(t_aug)
    if lam_eff:
        # J(t) = ∫ C_eff(t') * H(t') dt'
        for i in range(1, len(t_aug)):
            dt_i = t_aug[i] - t_aug[i - 1]
            I_aug[i] = I_aug[i - 1] + 0.5 * (
                C_eff_aug[i] * H_aug[i] + C_eff_aug[i - 1] * H_aug[i - 1]
            ) * dt_i
    else:
        # I(t) = ∫ H(t') dt'
        for i in range(1, len(t_aug)):
            dt_i = t_aug[i] - t_aug[i - 1]
            I_aug[i] = I_aug[i - 1] + 0.5 * (H_aug[i] + H_aug[i - 1]) * dt_i

    I = I_aug[1:] if t_sec[0] > 0.0 else I_aug

    I = np.maximum(I, 1e-100)
    H = np.maximum(H, 1e-100)
    # --- front ---
    if lam_eff:
        # xF^2 = pref * H^{-eps} * ∫ C_eff*H dt
        xF2 = pref * (H ** (-eps)) * I
    else:
        xF2 = pref * C0 * (H ** (-eps)) * I

    xF = np.sqrt(np.maximum(xF2, 0.0)) / 1.02  # adjust factor to match numerics

    # restore original order
    return WavefrontHelpers.restore_original_order(xF, order, t_sec_in.size)

# ============================================================================
# SECTION 2: MARSHAK BOUNDARY ITERATION (APPENDIX A) - CORE ENGINE
# ============================================================================

def _compute_Z1_from_C(eps, C_here, rho_here):
    return (f ** 2) * (rho_here ** (2.0 * (1.0 - mu))) * (2.0 + eps) * (1.0 - eps) * C_here


def _compute_ablation_step(i, t_sec, dt_i, t_heat, Ts_prev, xF_prev, wall_material, rho_here, data_of_R):
    global _ABLATION_R_ARRAY
    dE_wall_hJ = 0.0
    if wall_material == 'Copper':
        csv_path = U_TILDA_DIR / "u_tilda_copper(rho)_464_5.csv"
        u_tilde = AblationModel.get_u_tilda_closest(csv_path, rho_here)
    elif wall_material == 'Gold':
        csv_path = U_TILDA_DIR / "u_tilda_gold(rho)_510.1.csv"
        u_tilde = AblationModel.get_u_tilda_closest(csv_path, rho_here)
    else:
        u_tilde = None

    if wall_material in ('Gold', 'Copper'):
        if i > 1:
            R_array_prev = _ABLATION_R_ARRAY.copy()
        else:
            R_array_prev = None
        R_array = AblationModel.compute_R_t(t_sec[i], dt_i, t_heat, R_cm, Ts_prev, R_array_prev, wall_material=wall_material, u_tilde=u_tilde,)
    else:
        R_array = np.full_like(z, R_cm)

    _ABLATION_R_ARRAY = R_array
    data_of_R[t_sec[i]] = R_array.copy()
    A_i = np.pi * R_array[0] ** 2

    if wall_material in ('Gold', 'Copper'):
        dE_wall_hJ = WallLossModel.compute_wall_energy_loss(t_sec[i],dt_i,t_heat,R_cm,Ts_prev,xF_prev,flat_top_profile=True,wall=wall_material,)

    return R_array, A_i, dE_wall_hJ


_ABLATION_R_ARRAY = np.full_like(z, R_cm)


def _update_flux_and_energy(i, dt_i, base_flux, area, ablation, wall_loss, A, F, E, E_wall_erg, E_wall_array_erg, dE_wall_hJ):
    if ablation:
        E_wall_erg += dE_wall_hJ * 1e9
        E_wall_array_erg[i] = E_wall_erg
        F[i] = base_flux - (dE_wall_hJ * 1e9) / (A[i] * dt_i)
        E[i] = E[i - 1] + 0.5 * (F[i] * A[i] + F[i - 1] * A[i - 1]) * dt_i / area
    elif wall_loss:
        # Convert wall energy loss from hJ to erg and subtract from flux
        E_wall_erg += dE_wall_hJ * 1e9
        E_wall_array_erg[i] = E_wall_erg
        F[i] = base_flux - (dE_wall_hJ * 1e9) / (area * dt_i)
        E[i] = E[i - 1] + 0.5 * (F[i] + F[i - 1]) * dt_i
    else:
        # No ablation, no wall loss: just integrate the base flux
        E_wall_array_erg[i] = E_wall_erg
        F[i] = base_flux
        E[i] = E[i - 1] + 0.5 * (F[i] + F[i - 1]) * dt_i
    return E_wall_erg


def _update_vary_rho_terms(i, dt_i, Ts_prev, xF_prev, R_array, t_sec, p, K, eps, In, C_changing_rho, Z1_changing_rho, new_rho,  lam_eff, power, R_average_for_lambda_geom, g_eff_array, lambda_eff_array, lambda_R_array):
    new_rho[i] = AblationModel.compute_rho_effective(R_cm, R_array, xF_prev)

    if not lam_eff:
        In[i] = In[i - 1] + dt_i / 2 * (new_rho[i] ** p + new_rho[i - 1] ** p)
        C_changing_rho[i] = (K / t_sec[i]) * In[i]
    else:
        lambda_ross = g * (Ts_prev ** alpha) * (rho ** (-lambda_param - 1))
        xF_index = np.searchsorted(z, xF_prev)
        if R_average_for_lambda_geom:
            R_average = np.mean(R_array[:xF_index + 1]) if xF_index > 0 else R_array[0]
        else:
            R_average = R_cm
        lambda_geom = 2 * R_average
        lambda_eff = ((lambda_geom ** (-power) + lambda_ross ** (-power))) ** (-1 / power)
        lambda_eff_array[i] = lambda_eff
        lambda_R_array[i] = lambda_ross
        g_eff = g * (lambda_eff / lambda_ross)
        g_eff_array[i] = g_eff
        w_i = g_eff_array[i] / g
        w_im1 = g_eff_array[i - 1] / g
        In[i] = In[i - 1] + 0.5 * dt_i * (w_i * (new_rho[i] ** p) + w_im1 * (new_rho[i - 1] ** p))
        C_changing_rho[i] = (K / t_sec[i]) * In[i]

    Z1_changing_rho[i] = _compute_Z1_from_C(eps, C_changing_rho[i], new_rho[i])


def _update_wall_penetration_profiles(i, t_sec, dt_i, t_heat, Ts_i, xF_i, wall_material, wall_penetration_depth_cm_profile, vary_rho,):
    """Update cumulative wall penetration and map it to geometric gold radial grid.

    The wall front is measured from fixed foam boundary R_cm to R_cm + w_Au,
    while the ablation interface profile is tracked separately for plotting.
    """
    #xF_i is the current front position at r=R_cm.
    delta_penetration_depth_cm = WallLossModel.compute_wall_front_profile(t_sec[i], dt_i, t_heat, Ts_i, xF_i, flat_top_profile=True, wall=wall_material,)
    delta_penetration_depth_cm = np.asarray(delta_penetration_depth_cm, dtype=float)

    if delta_penetration_depth_cm.shape == wall_penetration_depth_cm_profile.shape:
        # we can directly add the new penetration depth to the what we have so far
        wall_penetration_depth_cm_profile += np.clip(delta_penetration_depth_cm, 0.0, None)
        wall_penetration_depth_cm_profile = np.clip(wall_penetration_depth_cm_profile, 0.0, w_Au)

        # print(f"wall material: {wall_material}, Total penetration depth (cm) = {wall_penetration_depth_cm_profile[1]:.3e}")

        # Enforce physically consistent shape for z <= xF_i:
        # penetration should be non-increasing with depth.
        heated_end = int(np.searchsorted(z, xF_i, side='right'))
        if heated_end > 1:
            heated_profile = np.clip(wall_penetration_depth_cm_profile[:heated_end], 0.0, w_Au)
            # Use an upper-envelope projection so near-surface values are not depressed by noise.
            heated_profile_monotone = np.maximum.accumulate(heated_profile[::-1])[::-1]
            wall_penetration_depth_cm_profile[:heated_end] = np.clip(heated_profile_monotone, 0.0, w_Au)

    # Wall-front location is referenced to the fixed foam boundary R_cm.
    wall_front_radius_profile = np.minimum(
        R_cm + wall_penetration_depth_cm_profile,
        r_gold[-1],
    )

    wall_penetration_radius_profile = np.minimum(
        wall_front_radius_profile,
        r_gold[-1], # don't allow penetration beyond the end of the gold grid
    )
    wall_penetration_cell_idx_profile = np.searchsorted(
        r_gold,
        wall_penetration_radius_profile,
        side='right',
    ) - 1
    wall_penetration_cell_idx_profile = np.clip(
        wall_penetration_cell_idx_profile,
        0,
        len(r_gold) - 1,
    )

    return (
        wall_penetration_depth_cm_profile,
        wall_penetration_radius_profile,
        wall_penetration_cell_idx_profile,
    )


def _update_shock_profiles(i, t_sec, dt_i, t_heat, Ts_i, xF_i, wall_material, wall_penetration_radius_profile):
    """Compute the shock penetration profile from the wall front.
    
    The shock front extends from the wall penetration front (already-penetrated wall interface). 
    Both shock_penetration_depth and shock_front_radius are 1D arrays (one value per depth z).
    """
    if wall_material != 'Gold':
        shock_penetration_depth_cm_profile = np.zeros_like(z, dtype=float)
        shock_front_radius_profile = np.asarray(wall_penetration_radius_profile, dtype=float).copy()
    else:
        # Compute shock penetration depth profile on z-grid
        shock_penetration_depth_cm_profile = WallLossModel.compute_shock_front_profile(
            t_sec[i],
            dt_i,
            t_heat,
            Ts_i,
            xF_i,
            wall=wall_material,
        )
        shock_penetration_depth_cm_profile = np.asarray(shock_penetration_depth_cm_profile, dtype=float)
        shock_penetration_depth_cm_profile = np.clip(shock_penetration_depth_cm_profile, 0.0, None)

        # Compute shock front radius profile: r_shock(z) = wall_front(z) + shock_depth(z)
        # The shock extends from the wall penetration front into the gold region
        wall_front = np.asarray(wall_penetration_radius_profile, dtype=float)
        shock_front_radius_profile = np.minimum(
            wall_front + shock_penetration_depth_cm_profile,
            r_gold[-1],
        )

    shock_penetration_cell_idx_profile = np.searchsorted(
        r_gold,
        shock_front_radius_profile,
        side='right',
    ) - 1
    shock_penetration_cell_idx_profile = np.clip(
        shock_penetration_cell_idx_profile,
        0,
        len(r_gold) - 1,
    )

    return (
        shock_penetration_depth_cm_profile,
        shock_front_radius_profile,
        shock_penetration_cell_idx_profile,
    )


def _update_t_heat(xF_i, t_heat, t_i):
    for j in range(len(z)):
        if z[j] <= xF_i and t_heat[j] == np.inf:
            t_heat[j] = t_i

def _compute_z_front_radial_snapshot(z_front, j0_profile):
    """Compute z_F(r,t) from centerline front and radial J0 profile."""
    j0_center = j0_profile[0] if j0_profile[0] != 0 else 1.0
    # j0_center is zero beacuse J_0(0) = 1, but we want to avoid division by zero if the profile is zero at the center for some reason. In that case, we can just return z_front as is, since the radial profile would be flat.
    return float(z_front) * (j0_profile / j0_center)


def _store_bessel_snapshot(
    i,
    t_sec,
    T_input_i,
    Ts_surface_i,
    dt_i,
    xF_i,
    E_wall_array_erg,
    bessel_data,
    data_of_R=None,
    t_ref_sec=None,
    t_heat=None,
    wall_interface_radius_profile=None,
    wall_penetration_depth_cm_profile=None,
    wall_penetration_radius_profile=None,
    wall_penetration_cell_idx_profile=None,
    shock_penetration_depth_cm_profile=None,
    shock_penetration_radius_profile=None,
    shock_penetration_cell_idx_profile=None,
    wall_material='Gold',
    ablation=False,
    lam_eff=False,
    power=2,
    lambda_eff_val=None,
):
    dE_wall = E_wall_array_erg[i] - E_wall_array_erg[i - 1]
    if i <= 2:
        albedo = AlbedoModel.compute_albedo_step(Ts_surface_i, dE_wall, dt_i, xF_i)
    else:
        # Find closest time key in bessel_data to avoid floating-point key mismatch
        t_prev_ns = t_sec[i-1] * 1e9
        closest_key = min(bessel_data.keys(), key=lambda k: abs(k - t_prev_ns)) if bessel_data else None
        albedo = bessel_data[closest_key]['avg_albedo']
    albedo_old = AlbedoModel.compute_albedo_step(Ts_surface_i, dE_wall, dt_i, xF_i)
    
    lambda_ross = g * (Ts_surface_i ** alpha) * (rho ** (-lambda_param - 1))
    
    if lam_eff and Experiment == "French":
        if lambda_eff_val is not None and lambda_eff_val > 0:
            lambda_use = lambda_eff_val
        # else:
        #     if data_of_R is not None and t_sec[i] in data_of_R:
        #         R_array = data_of_R[t_sec[i]]
        #         xF_index = np.searchsorted(z, xF_i)
        #         R_average = np.mean(R_array[:xF_index + 1]) if xF_index > 0 else R_array[0]
        #     else:
        #         R_average = R_cm
        #     lambda_geom = 2.0 * R_average
        #     lambda_eff = ((lambda_geom ** (-power) + lambda_ross ** (-power))) ** (-1 / power)
        #     lambda_use = lambda_eff
    else:
        lambda_use = lambda_ross
        
    epsilon = 3 / 4 * (1 - albedo) * (1 / lambda_use) * R_cm
    #epsilon = 3 / 2 * (1 - albedo) / (1 + albedo) * (1 / lambda_use) * R_cm
    kappa_0 = kappa_roots(epsilon, R_cm, n_roots=1)[0]
    kappa_0_approx = np.sqrt(2 * epsilon) / R_cm
    if i == 340: 
        print(f"t: {t_sec[i]},kappa_0: {kappa_0}, kappa_0_approx: {kappa_0_approx}, wall: {wall_material}")
        print(albedo)
    
    # Compute z_F at r=R_cm (edge of foam) using kappa_0
    z_F_at_rcm = xF_i * special.j0(kappa_0 * R_cm)
    albedo_array, avg_albedo = AlbedoModel.compute_albedo_profile(t_sec[i], dt_i, t_heat, Ts_surface_i, z_F_at_rcm, wall_material= wall_material)
    J0_profile = special.j0(kappa_0 * r_grid)
    J0_profile_approx = special.j0(kappa_0_approx * r_grid)
    z_F_radial = _compute_z_front_radial_snapshot(xF_i, J0_profile)
    z_F_radial_approx = _compute_z_front_radial_snapshot(xF_i, J0_profile_approx)

    snapshot = {
        'r_grid': r_grid.copy(),
        'z_grid': z.copy(),
        'J0_profiles': J0_profile.copy(),
        'J0_profiles_approx': J0_profile_approx.copy(),
        'kappa_0': kappa_0,
        'kappa_0_approx': kappa_0_approx,
        'z_F_radial': z_F_radial.copy(),
        'z_F_radial_approx': z_F_radial_approx.copy(),
        'z_F_at_rcm': z_F_at_rcm,
        'epsilon': epsilon,
        'albedo_old': albedo_old,
        'albedo': albedo,
        'lambda_ross': lambda_ross,
        'T_input': T_input_i,
        'Ts_surface': Ts_surface_i,
    }

    snapshot['albedo_array'] = albedo_array
    snapshot['avg_albedo'] = avg_albedo

    # Classify foam vs wall cells once in solver and store for plotting.
    if data_of_R is not None and t_ref_sec is not None and len(data_of_R) > 0 and ablation:
        z_mesh = z
        r_mesh = r_grid
        R_mesh, _ = np.meshgrid(r_mesh, z_mesh)
        dummy_temperature = np.ones((z_mesh.size, r_mesh.size), dtype=float)
        T_masked_foam, contour_r, contour_z, T_masked_wall = AblationModel.mask_wall_cells_from_ablation(dummy_temperature, R_mesh, z_mesh, data_of_R, t_ref_sec,)
        snapshot['ablation_foam_mask'] = np.isfinite(T_masked_foam)
        snapshot['ablation_wall_mask'] = np.isfinite(T_masked_wall)
        snapshot['ablation_contour_r'] = None if contour_r is None else np.asarray(contour_r, dtype=float)
        snapshot['ablation_contour_z'] = None if contour_z is None else np.asarray(contour_z, dtype=float)

    if wall_penetration_depth_cm_profile is not None:
        snapshot['wall_penetration_depth_cm_profile'] = np.asarray(wall_penetration_depth_cm_profile, dtype=float)
    if wall_penetration_radius_profile is not None:
        snapshot['wall_penetration_radius_profile'] = np.asarray(wall_penetration_radius_profile, dtype=float)
    if wall_penetration_cell_idx_profile is not None:
        snapshot['wall_penetration_cell_idx_profile'] = np.asarray(wall_penetration_cell_idx_profile, dtype=int)
    if shock_penetration_depth_cm_profile is not None:
        snapshot['shock_penetration_depth_cm_profile'] = np.asarray(shock_penetration_depth_cm_profile, dtype=float)
    if shock_penetration_radius_profile is not None:
        snapshot['shock_penetration_radius_profile'] = np.asarray(shock_penetration_radius_profile, dtype=float)
    if shock_penetration_cell_idx_profile is not None:
        snapshot['shock_penetration_cell_idx_profile'] = np.asarray(shock_penetration_cell_idx_profile, dtype=int)
    if wall_penetration_depth_cm_profile is not None or wall_penetration_radius_profile is not None:
        snapshot['r_gold_grid'] = r_gold.copy()

    bessel_data[t_sec[i] * 1e9] = snapshot


def _restore_marshak_outputs(xF, Ts, E_total_hJ, E_wall_hJ_array, order, t_sec_in, data_of_R, t_sec):
    xF_out = WavefrontHelpers.restore_original_order(xF, order, t_sec_in.size) / 1.02
    Ts_out = WavefrontHelpers.restore_original_order(Ts, order, t_sec_in.size)
    E_out = WavefrontHelpers.restore_original_order(E_total_hJ, order, t_sec_in.size)
    Ew_out = WavefrontHelpers.restore_original_order(E_wall_hJ_array, order, t_sec_in.size)
    data_of_R_ns = {1e9 * t_sec_in[i]: data_of_R[t_sec[i]] for i in range(t_sec_in.size)}
    return xF_out, Ts_out, E_out, Ew_out, data_of_R_ns

def _marshak_appendixA_march(times_to_store,*, use_seconds=True, wall_loss=False, ablation=False,
 vary_rho=False, flat_top_profile=True, wall_material='Gold', lam_eff=False, power=2, R_average_for_lambda_geom=True,):
    """
    Shared engine for Marshak boundary iteration (Appendix A).
    March in time:
    - Use Eq. (12) with Ts(t-dt) to compute flux:
        sigmaSB*TD^4(t) = sigmaSB*Ts^4(t-dt) + F(0,t)/2
    =>  F(0,t) = 2*sigmaSB*(TD^4(t) - Ts^4(t-dt))
    - Integrate flux to get E(t) (time integral over the flux)
    - Solve Eq. (A.3) implicitly for H(t)=Ts(t)^(4+alpha)
    - Compute xF(t) from Eq. (9) using this H(t)
    wall_loss: include compute_wall_energy_loss subtraction from flux
    ablation:  use compute_R_t + A_n = pi*R_array[0]^2 and flat_top_profile=True typically
    vary_rho:  compute new_rho and integrate C_changing_rho like your code
    flat_top_profile: passed into compute_wall_energy_loss
    Returns:
      (xF, Ts, E_total_hJ, E_wall_hJ_array)
    """
    t_sec, order, t_sec_in = WavefrontHelpers.prepare_times(times_to_store, use_seconds=use_seconds)
    if t_sec.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if t_sec[-1] > 1e-5:
        t_sec = t_sec * 1e-9

    eps, sigma_SB_hev, C0, pref = WavefrontHelpers.compute_constants_for_wavefront()

    # Heating-time per spatial zone (globals: z)
    t_heat = np.full_like(z, np.inf)
    
    # storage
    xF = np.zeros_like(t_sec)
    T_modelin = np.zeros_like(t_sec)
    H  = np.zeros_like(t_sec)
    I  = np.zeros_like(t_sec)
    F  = np.zeros_like(t_sec)
    E  = np.zeros_like(t_sec)  # "energy-like" integral over flux (per area-ish, matching your current usage)
    z_F_rcm = np.zeros_like(t_sec)  # z_F at r=R_cm (edge of foam)
    
    # 1D-only storage for albedo calculations
    Ts = np.zeros_like(t_sec)
    H_s  = np.zeros_like(t_sec)
    I_s  = np.zeros_like(t_sec)
    F_s  = np.zeros_like(t_sec)
    E_s  = np.zeros_like(t_sec)
    
    new_rho = np.full_like(t_sec, rho, dtype=float)
    C_changing_rho = np.full_like(t_sec, 0, dtype=float)
    Z1_changing_rho = np.full_like(t_sec, 0, dtype=float)
    Z1_changing_rho[0] = _compute_Z1_from_C(eps, C0, rho)
    A = np.full_like(t_sec, np.pi * R_cm ** 2, dtype=float)
    C_changing_rho[0] = C0
    In = np.full_like(t_sec, 0, dtype=float)
    Iw = np.zeros_like(t_sec)   # integral of w(t)*rho^p
    #In[0] = rho ** (mu+lambda_param-2)
    # wall energy (erg) cumulative, stored as array
    E_wall_erg = 0.0
    E_wall_array_erg = np.zeros_like(t_sec)
    # init
    # NOTE: you had Ts[0] = 0.62*get_TD(t_sec[0], ...). But get_TD expects ns in your code later.
    # I keep your original factor, but fix time units consistently: pass ns.
    T_modelin[0] = 0.01
    H[0]  = T_modelin[0] ** (4.0 + alpha)
    I[0]  = 0.0
    F[0]  = 0.0
    E[0]  = 0.0
    xF[0] = 0.0
    
    Ts[0] = 0.01
    H_s[0]  = Ts[0] ** (4.0 + alpha)
    I_s[0]  = 0.0
    F_s[0]  = 0.0
    E_s[0]  = 0.0
    g_eff_array = np.full_like(t_sec, g, dtype=float)
    lambda_eff_array = np.full_like(t_sec, 0, dtype=float)
    lambda_R_array = np.full_like(t_sec, 0, dtype=float)
    E_wall_array_erg[0] = 0.0
    data_of_R = {t: np.full_like(z, R_cm) for t in t_sec}  # store R(t) data if ablation
    wall_penetration_depth_cm_profile = np.zeros_like(z, dtype=float)
    shock_penetration_depth_cm_profile = np.zeros_like(z, dtype=float)
    shock_penetration_radius_profile = np.full_like(z, R_cm, dtype=float)
    shock_penetration_cell_idx_profile = np.zeros_like(z, dtype=int)
    
    # Bessel function data storage for each time step
    # Dictionary to store J_0(kappa_0 * r) profiles for each (time, z) pair
    bessel_data = {}  # keys: time (ns), values: dict with 'r_grid', 'J0_profiles' (shape: [Nz, Nr])

    area = np.pi * R_cm ** 2  # global R_cm

    #if vary_rho: 
    K = (16.0 / (4.0 + alpha)) * (g * sigma_SB_hev) / (3.0 * f)
    p = mu - lambda_param - 2.0

    # initial Z1
    Z1 = _compute_Z1_from_C(eps, C0, rho)
    R_array = np.full_like(z, R_cm)
    global _ABLATION_R_ARRAY # store previous R_array for ablation step
    _ABLATION_R_ARRAY = R_array.copy()
    for i in range(1, len(t_sec)):
        dt_i = t_sec[i] - t_sec[i - 1]
        if dt_i <= 0.0:
            dt_i = 0.0
        TD_now = get_TD(t_sec[i] * 1e9, t_array_TD, T_array_TD)
        T_modelin_prev = T_modelin[i - 1]
        C_eff = C0
        
        # Always run the 1D model to obtain Ts_1D for albedo calculations
        base_flux_1D = 2.0 * sigma_SB_hev * (TD_now**4 - Ts[i - 1]**4)
        F_s[i] = base_flux_1D
        E_s[i] = E_s[i - 1] + 0.5 * (F_s[i] + F_s[i - 1]) * dt_i
        Z1_1D = Z1_changing_rho[0]  # Z1 for constant C0 and constant rho
        H_1D_new = WavefrontHelpers.solve_for_H_new_brentq(Z1_1D, eps, E_s[i]**2, I_s[i - 1], H_s[i - 1], dt_i)
        H_s[i] = max(H_1D_new, 1e-100)
        Ts[i] = H_s[i] ** (1.0 / (4.0 + alpha))
        I_s[i] = I_s[i - 1] + 0.5 * (H_s[i - 1] + H_s[i]) * dt_i

        if lam_eff and (not vary_rho):
            lambda_ross = g*(Ts[i-1]**alpha)*(rho**(-lambda_param-1))
            lambda_geom = 2*R_cm
            lambda_eff = ((lambda_geom**(-power) + lambda_ross**(-power)))**(-1/power)
            lambda_eff_array[i] = lambda_eff
            lambda_R_array[i] = lambda_ross
            g_eff = g * lambda_eff / lambda_ross
            g_eff_array[i] = g_eff
             # weight for the integral
            w_i   = g_eff_array[i]   / g
            w_im1 = g_eff_array[i-1] / g

            # integrate w * rho^p
            Iw[i] = Iw[i-1] + 0.5 * dt_i * (w_i + w_im1) * (rho**p)

            # now C_eff comes from the integral (not relaxation)
            C_eff = (K / t_sec[i]) * Iw[i]

            Z1 = _compute_Z1_from_C(eps, C_eff, rho)
        # ---- Compute flux (Eq. 12) and integrate to E ----
        # Base Marshak flux:
        base_flux = 2.0 * sigma_SB_hev * (TD_now**4 - Ts[i-1]**4)

        # Default values for ablation geometry
        A[i] = area  # fallback; you sometimes scale with A_n/area
        dE_wall_hJ = 0.0

        if ablation:
            R_array, A[i], dE_wall_hJ = _compute_ablation_step(i, t_sec, dt_i, t_heat, Ts[i-1], z_F_rcm[i-1], wall_material, rho, data_of_R,)
        elif wall_loss:
            dE_wall_hJ = WallLossModel.compute_wall_energy_loss(t_sec[i], dt_i, t_heat, R_cm, Ts[i-1], z_F_rcm[i-1], flat_top_profile=Flattop_condition, wall=wall_material,)

        E_wall_erg = _update_flux_and_energy(i, dt_i,base_flux, area,ablation, wall_loss, A, F, E, E_wall_erg, E_wall_array_erg, dE_wall_hJ,)

        # ---- If vary_rho: update rho_eff and C_changing_rho and Z1 ----
        if vary_rho and ablation:
            _update_vary_rho_terms(i, dt_i, Ts[i-1], z_F_rcm[i-1], R_array, t_sec, p, K, eps, In, C_changing_rho, Z1_changing_rho, new_rho, lam_eff, power, R_average_for_lambda_geom, g_eff_array,lambda_eff_array,lambda_R_array,)
        
        # ---- Solve Eq. (A.3) for H ----
        E2 = E[i] ** 2
        I_prev = I[i - 1]
        H_prev = H[i - 1]
        if vary_rho and ablation:
            Z1 = Z1_changing_rho[i]
        H_new = WavefrontHelpers.solve_for_H_new_brentq(Z1, eps, E2, I_prev, H_prev, dt_i)

        H[i] = max(H_new, 1e-100)
        Ts[i] = H[i] ** (1.0 / (4.0 + alpha))
        I[i] = I_prev + 0.5 * (H_prev + H[i]) * dt_i

        # ---- Update xF and heating times ----
        C_use = C_changing_rho[i] if vary_rho else C0
        if lam_eff and not vary_rho:
            C_use = C_eff
        xF2 = pref * C_use * (H[i] ** (-eps)) * I[i]
        xF[i] = np.sqrt(np.maximum(xF2, 0.0))
        z_F_rcm[i] = xF[i]
        (wall_penetration_depth_cm_profile, wall_penetration_radius_profile, wall_penetration_cell_idx_profile,) = _update_wall_penetration_profiles(
                i, t_sec, dt_i, t_heat, Ts[i], z_F_rcm[i-1], wall_material, wall_penetration_depth_cm_profile, vary_rho,)
        (shock_penetration_depth_cm_profile, shock_penetration_radius_profile, shock_penetration_cell_idx_profile,) = _update_shock_profiles(
            i, t_sec, dt_i, t_heat, Ts[i], z_F_rcm[i-1], wall_material, wall_penetration_radius_profile,)
        if wall_loss and i > 1:
            #calcultates the bessel function profiles and parameters for the current time step and stores them in bessel_data for later retrieval and plotting
            _store_bessel_snapshot(i, t_sec, T_modelin[i], Ts[i], dt_i, xF[i], E_wall_array_erg, bessel_data, data_of_R=data_of_R if ablation else None, t_ref_sec=t_sec[i] if ablation else None, t_heat=t_heat, wall_penetration_depth_cm_profile=wall_penetration_depth_cm_profile, wall_penetration_radius_profile=wall_penetration_radius_profile, wall_penetration_cell_idx_profile=wall_penetration_cell_idx_profile, shock_penetration_depth_cm_profile=shock_penetration_depth_cm_profile, shock_penetration_radius_profile=shock_penetration_radius_profile, shock_penetration_cell_idx_profile=shock_penetration_cell_idx_profile,
                                   wall_material=wall_material, ablation=ablation, lam_eff=lam_eff, power=power, lambda_eff_val=lambda_eff_array[i])

            # Extract z_F at r=R_cm from the current time snapshot.
            t_key = t_sec[i] * 1e9
            if t_key in bessel_data and 'z_F_at_rcm' in bessel_data[t_key]:
                z_F_rcm[i] = bessel_data[t_key]['z_F_at_rcm']
                # z_F_rcm[i] = xF[i]
        _update_t_heat(z_F_rcm[i], t_heat, t_sec[i])

        #calculating t_heat - according to the definition of t_heat, we want to find the time at which each spatial zone z is first reached by the heat front. The heat front position at r=R_cm is given by z_F_rcm[i], so we check for each spatial zone z[j] whether it has been reached by the heat front (z[j] <= z_F_rcm[i]) and if it hasn't been reached before (t_heat[j] == np.inf). If both conditions are true, we update t_heat[j] to the current time t_sec[i].            
    # Convert energies to hJ like your return:
    # You returned: E*1e-9*area, E_wall_array*1e-9
    # That's (erg/cm^2)*area -> erg ; *1e-9 -> hJ (since 1 hJ = 1e-7 J = 1e0 erg? careful)
    # I will keep EXACTLY your conversion factors to preserve your downstream plots.
    E_total_hJ = E * 1e-9 * area
    E_wall_hJ_array = E_wall_array_erg * 1e-9

    #plot lamda_r, and lam_eff as a function of time (also put the 2*R_cm)
    if lam_eff:
        # Calculate lambda_geom over time
        lambda_geom = np.zeros_like(t_sec)
        for idx in range(len(t_sec)):
            if R_average_for_lambda_geom and ablation and (t_sec[idx] in data_of_R):
                xF_index = np.searchsorted(z, xF[idx])
                R_average = np.mean(data_of_R[t_sec[idx]][:xF_index + 1]) if xF_index > 0 else data_of_R[t_sec[idx]][0]
                lambda_geom[idx] = 2.0 * R_average
            else:
                lambda_geom[idx] = 2.0 * R_cm

        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_eff_1 = (lambda_geom**(-1.0) + lambda_R_array**(-1.0))**(-1.0)
            lambda_eff_2 = (lambda_geom**(-2.0) + lambda_R_array**(-2.0))**(-0.5)
            lambda_eff_5 = (lambda_geom**(-5.0) + lambda_R_array**(-5.0))**(-0.2)

        # Convert t_sec to ns for plotting if it is in seconds
        t_ns_plot = t_sec * 1e9 if t_sec[-1] < 1e-5 else t_sec

        plt.figure(figsize=(8, 6))
        plt.plot(t_ns_plot[1:], lambda_eff_1[1:], label=r"$\lambda_{\mathrm{eff}}$ ($n=1$)", linestyle="-")
        plt.plot(t_ns_plot[1:], lambda_eff_2[1:], label=r"$\lambda_{\mathrm{eff}}$ ($n=2$)", linestyle="-")
        plt.plot(t_ns_plot[1:], lambda_eff_5[1:], label=r"$\lambda_{\mathrm{eff}}$ ($n=5$)", linestyle="-")
        plt.plot(t_ns_plot[1:], lambda_R_array[1:], label=r"$\lambda_{\mathrm{R}}$ (Rosseland)", linestyle="--", color='green')
        plt.plot(t_ns_plot[1:], lambda_geom[1:], label=r"$2R$ (Geometric)", linestyle="--", color='red')
        plt.xlabel(r"$t$ [ns]", fontsize=18, fontname='serif')
        plt.ylabel(r"$\lambda$ [cm]", fontsize=18, fontname='serif')
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.legend(prop={'family': 'serif', 'size': 16})
        plt.tight_layout()
        try:
            from csv_helpers import save_figure
            save_figure("lambda_eff_vs_time.png", model1_5=True)
        except Exception as e:
            print(f"Warning: could not save lambda_eff plot: {e}")
        plt.show()

    xF_out, Ts_out, E_out, Ew_out, data_of_R = _restore_marshak_outputs(xF, Ts, E_total_hJ, E_wall_hJ_array, order, t_sec_in, data_of_R, t_sec,)
    A_arr, T_lead_out, T_corr_out = WavefrontHelpers.compute_first_order_hr_profile(t_sec, xF, Ts)
    if isinstance(bessel_data, dict):
        bessel_data['profile_A_t'] = WavefrontHelpers.restore_original_order(A_arr, order, t_sec_in.size)
        bessel_data['T_profile_leading'] = T_lead_out
        bessel_data['T_profile_corrected'] = T_corr_out
    return xF_out, Ts_out, E_out, Ew_out, data_of_R, bessel_data


# ============================================================================
# SECTION 2B: COMPACT BATH-TEMPERATURE MARSHAK MODEL (FLUENCE-RATIO METHOD)
# ============================================================================
# Reference: HR_corrected.tex, Sections 1–7
# This is a standalone alternative to the Appendix A time-marching method.
# It pre-computes bath fluences and solves a scalar algebraic equation for
# u(t) = (Ts/Tb)^4 at each time.


def analytic_wave_front_marshak_fluence(times_to_store, *, use_seconds=True, k=10):
    """
        Bath-temperature HR Marshak model with successive global history corrections.

        Instead of Appendix A implicit time marching, this method:
            1. Pre-computes bath fluences J_4(t) and J_n(t) by trapezoidal integration.
            2. Solves the zeroth-order scalar equation for u^(0)(t) = (Ts/Tb)^4.
            3. Applies k global-history correction steps to obtain u^(k)(t).
            4. Computes x_F, T_s, and spatial temperature profiles from Ts^(k)(t).

    Parameters
    ----------
    times_to_store : array-like
        Times at which to compute results (ns if use_seconds=False, seconds if True).
    use_seconds : bool
        If True, times_to_store is in seconds; if False, in nanoseconds.
    k : int
        Number of successive global history-correction steps (K in the derivation).
        k=0 recovers the compact fluence model. Default is 10.

    Returns
    -------
    xF : np.ndarray, shape (N_times,)
        Heat-front position in cm.
    Ts : np.ndarray, shape (N_times,)
        Surface temperature in HeV (hundreds of eV).
    T_profile_leading : np.ndarray, shape (N_times, N_x)
        Leading-order temperature profile T(x,t) = Ts*(1 - x/xF)^(1/n)
        on the global z grid.
    T_profile_corrected : np.ndarray, shape (N_times, N_x)
        First-order corrected HR temperature profile (Eq. 22 of HR_corrected.tex)
        on the global z grid.
    """

    # ---- Prepare time grid ----
    t_sec, order, t_sec_in = WavefrontHelpers.prepare_times(
        times_to_store, use_seconds=use_seconds
    )
    N = t_sec.size
    N_x = z.size

    if N == 0:
        return (np.array([]), np.array([]),
                np.empty((0, N_x)), np.empty((0, N_x)))

    # If times are in ns (> 1e-5), convert to seconds
    if t_sec[-1] > 1e-5:
        t_sec = t_sec * 1e-9

    # ---- Material constants (Section 1) ----
    eps, sigma_SB_hev, C, pref = WavefrontHelpers.compute_constants_for_wavefront()
    n = 4.0 + alpha       # Eq. (3): n = 4 + alpha
    # eps = beta / n       # already computed in WavefrontHelpers
    K_eps = pref           # pref = (2 + eps) / (1 - eps) = K_eps  (Eq. 6)

    # Z1 = f^2 * rho^{2(1-mu)} * (2+eps) * (1-eps) * C   (Eq. 12)
    Z1 = _compute_Z1_from_C(eps, C, rho)

    # ---- Interpolate T_b(t) in HeV ----
    # t_array_TD is in ns, T_array_TD is in eV; get_TD converts to HeV (factor 0.01)
    t_ns = t_sec * 1e9
    Tb_hev = np.array([get_TD(t_ns[i], t_array_TD, T_array_TD) for i in range(N)])

    # ---- Compute bath fluences J_4 and J_n (Section 4, Eq. 16) ----
    # Augment with t=0 if needed for proper integration from zero
    if t_sec[0] > 0.0:
        t_aug = np.concatenate(([0.0], t_sec))
        Tb0 = get_TD(0.0, t_array_TD, T_array_TD)
        Tb_aug = np.concatenate(([Tb0], Tb_hev))
    else:
        t_aug = t_sec.copy()
        Tb_aug = Tb_hev.copy()

    Tb4_aug = Tb_aug ** 4
    Tbn_aug = Tb_aug ** n

    # Trapezoidal cumulative integration
    J4_aug = np.zeros_like(t_aug)
    Jn_aug = np.zeros_like(t_aug)
    for i in range(1, len(t_aug)):
        dt_i = t_aug[i] - t_aug[i - 1]
        J4_aug[i] = J4_aug[i - 1] + 0.5 * (Tb4_aug[i] + Tb4_aug[i - 1]) * dt_i
        Jn_aug[i] = Jn_aug[i - 1] + 0.5 * (Tbn_aug[i] + Tbn_aug[i - 1]) * dt_i

    # Extract values corresponding to the original time points
    if t_sec[0] > 0.0:
        J4 = J4_aug[1:]
        Jn = Jn_aug[1:]
    else:
        J4 = J4_aug
        Jn = Jn_aug

    # ---- Solve for u(t): zeroth-order + k global history corrections ----
    try:
        k_iter = int(k)
    except Exception as exc:
        raise ValueError("k must be an integer >= 0") from exc
    if k_iter < 0:
        raise ValueError("k must be >= 0")

    exponent_u = (n + beta) / 8.0   # power of u in algebraic equations
    u_arr = np.zeros(N)
    xF_arr = np.zeros(N)
    Ts_arr = np.zeros(N)
    T_profile_leading = np.zeros((N, N_x))
    T_profile_corrected = np.zeros((N, N_x))

    valid = (Tb_hev > 1e-30) & (J4 > 1e-100) & (Jn > 1e-100)

    # Base coefficient: sqrt(Z1*Tb^beta*Jn / (2*sigma*J4)^2)
    base_coeff = np.zeros(N)
    denom = 2.0 * sigma_SB_hev * J4
    safe = valid & (denom > 1e-200)
    base_coeff[safe] = np.sqrt(Z1 * (Tb_hev[safe] ** beta) * Jn[safe]) / denom[safe]

    # Zeroth-order solve: 1 - u = base_coeff * u^exp
    for i in range(N):
        if not valid[i]:
            u_arr[i] = 0.0
            continue
        coeff = base_coeff[i]

        def Phi(u, _coeff=coeff, _exp=exponent_u):
            return 1.0 - u - _coeff * (u ** _exp)

        if coeff <= 0.0:
            u_arr[i] = 1.0
        else:
            try:
                u_i = scipy.optimize.brentq(Phi, 1e-15, 1.0 - 1e-15, maxiter=200)
            except ValueError:
                u_i = 0.5
            u_arr[i] = u_i

    # Successive global history corrections:
    # 1 - I4^(k) * u^(k+1) = sqrt(base_coeff^2 * In^(k)) * (u^(k+1))^exp
    for _ in range(k_iter):
        u_prev = np.maximum(u_arr.copy(), 1e-18)

        # Integrate history on a grid augmented with t=0 to avoid
        # near-zero first-step artifacts in I4/In factors.
        if t_sec[0] > 0.0:
            Tb0 = get_TD(0.0, t_array_TD, T_array_TD)
            t_hist = np.concatenate(([0.0], t_sec))
            Tb_hist = np.concatenate(([Tb0], Tb_hev))
            u_hist = np.concatenate(([u_prev[0]], u_prev))
        else:
            t_hist = t_sec
            Tb_hist = Tb_hev
            u_hist = u_prev

        I4_hist = np.zeros_like(t_hist)
        In_hist = np.zeros_like(t_hist)
        u_hist_n4 = u_hist ** (n / 4.0)

        for i in range(1, len(t_hist)):
            dt_i = t_hist[i] - t_hist[i - 1]
            if dt_i <= 0.0:
                continue
            I4_hist[i] = I4_hist[i - 1] + 0.5 * dt_i * (
                u_hist[i] * (Tb_hist[i] ** 4) + u_hist[i - 1] * (Tb_hist[i - 1] ** 4)
            )
            In_hist[i] = In_hist[i - 1] + 0.5 * dt_i * (
                u_hist_n4[i] * (Tb_hist[i] ** n) + u_hist_n4[i - 1] * (Tb_hist[i - 1] ** n)
            )

        if t_sec[0] > 0.0:
            I4_int = I4_hist[1:]
            In_int = In_hist[1:]
        else:
            I4_int = I4_hist
            In_int = In_hist
        u_prev_n4 = u_prev ** (n / 4.0)

        I4_fac = np.ones(N)
        In_fac = np.ones(N)
        denom_I4 = np.maximum(u_prev * J4, 1e-30)
        denom_In = np.maximum(u_prev_n4 * Jn, 1e-30)
        I4_fac[valid] = I4_int[valid] / denom_I4[valid]
        In_fac[valid] = In_int[valid] / denom_In[valid]

        I4_fac = np.clip(I4_fac, 1e-12, np.inf)
        In_fac = np.clip(In_fac, 1e-12, np.inf)

        for i in range(N):
            if not valid[i]:
                u_arr[i] = 0.0
                continue

            I4_i = I4_fac[i]
            coeff_i = base_coeff[i] * np.sqrt(In_fac[i])

            # Monotone residual for Eq. (21)
            def Phi_hist(u, _I4=I4_i, _coeff=coeff_i, _exp=exponent_u):
                return 1.0 - _I4 * u - _coeff * (u ** _exp)

            u_hi = max(1e-12, (1.0 / I4_i) * (1.0 - 1e-12))
            if coeff_i <= 0.0:
                u_arr[i] = min(1.0 / I4_i, u_hi)
                continue

            try:
                u_arr[i] = scipy.optimize.brentq(Phi_hist, 1e-15, u_hi, maxiter=250)
            except ValueError:
                # Safe fallback on the guaranteed interval scale
                u_arr[i] = 0.5 * u_hi

    # ---- Compute Ts and xF from corrected Ts history (updated derivation Eq. 25) ----
    Ts_arr[valid] = Tb_hev[valid] * (u_arr[valid] ** 0.25)

    # Jn_Ts(t) = \int_0^t [Ts(t')]^n dt' using trapezoidal integration
    if t_sec[0] > 0.0:
        t_front = np.concatenate(([0.0], t_sec))
        Ts0 = get_TD(0.0, t_array_TD, T_array_TD) * (max(u_arr[0], 0.0) ** 0.25)
        Ts_front = np.concatenate(([Ts0], Ts_arr))
    else:
        t_front = t_sec
        Ts_front = Ts_arr

    Tsn_front = Ts_front ** n
    Jn_Ts_front = np.zeros_like(t_front)
    for i in range(1, len(t_front)):
        dt_i = t_front[i] - t_front[i - 1]
        if dt_i <= 0.0:
            continue
        Jn_Ts_front[i] = Jn_Ts_front[i - 1] + 0.5 * dt_i * (Tsn_front[i] + Tsn_front[i - 1])

    if t_sec[0] > 0.0:
        Jn_Ts = Jn_Ts_front[1:]
    else:
        Jn_Ts = Jn_Ts_front

    xF2 = np.zeros(N)
    Ts_safe = np.maximum(Ts_arr, 1e-30)
    xF2[valid] = K_eps * C * (Ts_safe[valid] ** (-beta)) * Jn_Ts[valid]
    xF_arr = np.sqrt(np.maximum(xF2, 0.0))

    # ---- Compute temperature profiles (Section 6) ----

    # Profile parameter A(t) (Eq. 23):
    # A = xF^2 / (C * Ts^{n(1-eps)}) * d/dt[ln(Ts^n)]
    # = xF^2 / (C * H^{1-eps}) * Hdot / H
    # where H = Ts^n, Hdot estimated by finite differences
    H_arr = Ts_arr ** n
    Hdot = np.zeros(N)
    for i in range(1, N):
        dt_i = t_sec[i] - t_sec[i - 1]
        if dt_i > 0:
            Hdot[i] = (H_arr[i] - H_arr[i - 1]) / dt_i
        else:
            Hdot[i] = 0.0
    # At i=0, use forward difference if possible
    if N > 1:
        dt0 = t_sec[1] - t_sec[0]
        if dt0 > 0:
            Hdot[0] = (H_arr[1] - H_arr[0]) / dt0

    A_arr = np.zeros(N)
    for i in range(N):
        H_i = H_arr[i]
        if H_i > 1e-100 and xF_arr[i] > 0 and C > 0:
            H_1meps = H_i ** (1.0 - eps)
            if H_1meps > 1e-100:
                A_arr[i] = (xF_arr[i] ** 2) / (C * H_1meps) * Hdot[i] / H_i

    for i in range(N):
        Ts_i = Ts_arr[i]
        xF_i = xF_arr[i]
        A_i = A_arr[i]

        if xF_i < 1e-30 or Ts_i < 1e-30:
            # Front hasn't moved; T = 0 everywhere
            continue

        y = z / xF_i  # normalized coordinate y = x / xF

        for j in range(N_x):
            if y[j] >= 1.0:
                # Beyond the front: T = 0
                T_profile_leading[i, j] = 0.0
                T_profile_corrected[i, j] = 0.0
            else:
                # Leading-order profile (Eq. 25):
                # T = Ts * (1 - y)^{1/n}
                T_profile_leading[i, j] = Ts_i * ((1.0 - y[j]) ** (1.0 / n))

                # First-order corrected profile (Eq. 22):
                # T = Ts * [(1-y)(1 + eps/2 * (1-A) * y)]^{1/[n(1-eps)]}
                inner = (1.0 - y[j]) * (1.0 + (eps / 2.0) * (1.0 - A_i) * y[j])
                if inner > 0:
                    T_profile_corrected[i, j] = Ts_i * (inner ** (1.0 / (n * (1.0 - eps))))
                else:
                    T_profile_corrected[i, j] = 0.0

    # ---- Restore original time ordering ----
    xF_out = WavefrontHelpers.restore_original_order(xF_arr, order, t_sec_in.size)
    Ts_out = WavefrontHelpers.restore_original_order(Ts_arr, order, t_sec_in.size)

    # Restore profile arrays to original time order
    T_lead_out = np.zeros((t_sec_in.size, N_x))
    T_corr_out = np.zeros((t_sec_in.size, N_x))
    for i_sorted in range(N):
        i_orig = order[i_sorted]
        T_lead_out[i_orig, :] = T_profile_leading[i_sorted, :]
        T_corr_out[i_orig, :] = T_profile_corrected[i_sorted, :]

    return xF_out, Ts_out, T_lead_out, T_corr_out


# ============================================================================
# SECTION 3: PUBLIC SOLVER INTERFACE
# ============================================================================

# --- Mode 2: Marshak (Appendix A) - no wall loss ---

def analytic_wave_front_marshak(times_to_store, *, use_seconds=True, wall_material='Gold', lam_eff=False, power=2):
    """
    Marshak boundary iteration (Appendix A), no wall loss.
    Returns: xF, Ts, E_total_hJ, E_wall_hJ_array
    """
    return _get_default_solver().analytic_wave_front_marshak(
        times_to_store,
        use_seconds=use_seconds,
        wall_material=wall_material,
        lam_eff=lam_eff,
        power=power,
    )

# --- Mode 3: Marshak + loss to gold wall (no ablation) ---

def analytic_wave_front_marshak_gold_loss(times_to_store, *, use_seconds=True, wall_material='Gold', lam_eff=False, power=2):
    """
    Marshak boundary iteration + wall loss to gold, no ablation.
    Returns: xF, Ts, E_total_hJ, E_wall_hJ_array
    """
    return _get_default_solver().analytic_wave_front_marshak_gold_loss(
        times_to_store,
        use_seconds=use_seconds,
        wall_material=wall_material,
        lam_eff=lam_eff,
        power=power,
    )

# --- Mode 4: Marshak + ablation (includes gold wall loss) ---

def analytic_wave_front_marshak_ablation(times_to_store, *, use_seconds=True, vary_rho=False, wall_material='Gold', lam_eff=False, power=2, R_average_for_lambda_geom=False):
    """
    Marshak boundary iteration + ablation + wall loss to gold.
    Returns: xF, Ts, E_total_hJ, E_wall_hJ_array
    """
    return _get_default_solver().analytic_wave_front_marshak_ablation(
        times_to_store,
        use_seconds=use_seconds,
        vary_rho=vary_rho,
        wall_material=wall_material,
        lam_eff=lam_eff,
        power=power,
        R_average_for_lambda_geom=R_average_for_lambda_geom,
    )

# --- Mode dispatcher: selectable entry point ---

def analytic_wave_front_dispatch(
    times_to_store,
    *,
    use_seconds=True,
    mode="no_marshak",
    vary_rho=False,
    wall_material='Gold',
    lam_eff=False,
    power=2,
    k=10,
    R_average_for_lambda_geom=False,
):
    """
    mode options:
      - "no_marshak"
      - "marshak"
      - "marshak_wall_loss"
      - "marshak_ablation"
      - "marshak_fluence"
    """
    return _get_default_solver().analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=use_seconds,
        mode=mode,
        vary_rho=vary_rho,
        wall_material=wall_material,
        lam_eff=lam_eff,
        power=power,
        k=k,
        R_average_for_lambda_geom=R_average_for_lambda_geom,
    )

######################################################################################

_DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER = None


def _get_default_solver():
    global _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER
    if _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER is None:
        _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER = AnalyticalWavefrontSolver(
            no_marshak_fn=analytic_wave_front_no_marshak,
            march_fn=_marshak_appendixA_march,
            fluence_fn=analytic_wave_front_marshak_fluence,
        )
    return _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER
