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
            TD_i = get_TD(t_sec[i]*1e9, t_array_TD, T_array_TD)
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
    Ts_hev = np.array([get_TD(ti, t_array_TD, T_array_TD) for ti in t_ns], dtype=float)
    H = Ts_hev ** (4.0 + alpha)

    # cumulative trapezoid I(t)=∫_0^t H dt
    if t_sec[0] > 0.0:
        t_aug = np.concatenate(([0.0], t_sec))
        H0 = get_TD(0.0, t_array_TD, T_array_TD) ** (4.0 + alpha)
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

def _marshak_appendixA_march(
    times_to_store,
    *,
    use_seconds=True,
    wall_loss=False,
    ablation=False,
    vary_rho=False,
    flat_top_profile=False,
    wall_material='Gold',
    lam_eff=False,
    power=2,
    R_average_for_lambda_geom=True,
    good_way=False
):
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
    if t_sec[-1] > 1e-5:
        t_sec = t_sec * 1e-9
    if t_sec.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    eps, sigma_SB_hev, C0, pref = WavefrontHelpers.compute_constants_for_wavefront()

    # Heating-time per spatial zone (globals: z)
    t_heat = np.full_like(z, np.inf)

    # Appendix A Z1 uses C (but we may update it if vary_rho=True)
    # Z1 = f^2 * rho^(2(1-mu)) * (2+eps)(1-eps) * C
    # Z2 = Z1 * ∫_0^{t-dt} H(t') dt'  = Z1 * I_prev
    def compute_Z1(C_here, rho_here):
        return (f ** 2) * (rho_here ** (2.0 * (1.0 - mu))) * (2.0 + eps) * (1.0 - eps) * C_here
    
    # storage
    xF = np.zeros_like(t_sec)
    Ts = np.zeros_like(t_sec)
    H  = np.zeros_like(t_sec)
    I  = np.zeros_like(t_sec)
    F  = np.zeros_like(t_sec)
    E  = np.zeros_like(t_sec)  # "energy-like" integral over flux (per area-ish, matching your current usage)

    new_rho = np.full_like(t_sec, rho, dtype=float)
    C_changing_rho = np.full_like(t_sec, 0, dtype=float)
    Z1_changing_rho = np.full_like(t_sec, 0, dtype=float)
    Z1_changing_rho[0] = compute_Z1(C0, rho)
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
    Ts[0] = 0.01
    H[0]  = Ts[0] ** (4.0 + alpha)
    I[0]  = 0.0
    F[0]  = 0.0
    E[0]  = 0.0
    xF[0] = 0.0
    g_eff_array = np.full_like(t_sec, g, dtype=float)
    lambda_eff_array = np.full_like(t_sec, 0, dtype=float)
    C_eff_array = np.full_like(t_sec, C0, dtype=float)
    E_wall_array_erg[0] = 0.0
    data_of_R = {t: np.full_like(z, R_cm) for t in t_sec}  # store R(t) data if ablation
    
    # Bessel function data storage for each time step
    # Create radial grid from 0.01 cm to R_cm
    n_r_points = 100
    r_grid = np.linspace(0.0, R_cm, n_r_points)
    # Dictionary to store J_0(kappa_0 * r) profiles for each (time, z) pair
    bessel_data = {}  # keys: time (ns), values: dict with 'r_grid', 'J0_profiles' (shape: [Nz, Nr])

    area = np.pi * R_cm ** 2  # global R_cm

    #if vary_rho: 
    K = (16.0 / (4.0 + alpha)) * (g * sigma_SB_hev) / (3.0 * f)
    p = mu - lambda_param - 2.0


    # initial Z1
    Z1 = compute_Z1(C0, rho)
    for i in range(1, len(t_sec)):
        dt_i = t_sec[i] - t_sec[i - 1]
        if dt_i <= 0.0:
            dt_i = 0.0
        TD_now = get_TD(t_sec[i] * 1e9, t_array_TD, T_array_TD)
        Ts_prev = Ts[i - 1]
        if lam_eff and (not vary_rho):
            lambda_ross = g*(Ts_prev**alpha)*(rho**(-lambda_param-1))
            lambda_geom = 2*R_cm
            lambda_eff = ((lambda_geom**(-power) + lambda_ross**(-power)))**(-1/power)
            lambda_eff_array[i] = lambda_eff
            g_eff = g * lambda_eff / lambda_ross
            g_eff_array[i] = g_eff
             # weight for the integral
            w_i   = g_eff_array[i]   / g
            w_im1 = g_eff_array[i-1] / g

            # integrate w * rho^p
            Iw[i] = Iw[i-1] + 0.5 * dt_i * (w_i + w_im1) * (rho**p)

            # now C_eff comes from the integral (not relaxation)
            C_eff = (K / t_sec[i]) * Iw[i]

            Z1 = compute_Z1(C_eff, rho)
        # ---- Compute flux (Eq. 12) and integrate to E ----
        # Base Marshak flux:
        base_flux = 2.0 * sigma_SB_hev * (TD_now**4 - Ts_prev**4)

        # Default values for ablation geometry
        A[i] = area  # fallback; you sometimes scale with A_n/area
        dE_wall_hJ = 0.0

        if ablation:
            # Ablation geometry + wall loss are coupled here in your code
            if wall_material == 'Cupper':
                csv_path = U_TILDA_DIR / "u_tilda_cupper(rho)_464_5.csv"
                u_tilde = AblationModel.get_u_tilda_closest(csv_path, rho)
            elif wall_material == 'Gold':
                csv_path = U_TILDA_DIR / "u_tilda_gold(rho)_510.1.csv"
                u_tilde = AblationModel.get_u_tilda_closest(csv_path, rho)
            if wall_material == 'Gold' or wall_material == 'Cupper':
                if i > 1:
                    R_array_prev = R_array.copy()
                    R_array = AblationModel.compute_R_t(t_sec[i], dt_i, t_heat, R_cm, Ts_prev, R_array_prev, wall_material=wall_material, u_tilde=u_tilde)  # [cm]
                else:
                    R_array_prev = None
                    R_array = AblationModel.compute_R_t(t_sec[i], dt_i, t_heat, R_cm, Ts_prev, R_array_prev, wall_material=wall_material, u_tilde=u_tilde)  # [cm]
            data_of_R[t_sec[i]] = R_array.copy()
            A[i] = np.pi * R_array[0] ** 2

            # your wall loss call in ablation branch
            if wall_material == 'Gold' or wall_material == 'Cupper':
                dE_wall_hJ = WallLossModel.compute_wall_energy_loss(t_sec[i], dt_i, t_heat, R_cm, Ts_prev, xF[i - 1],flat_top_profile=True, wall=wall_material) #flat top T always True for ablation
                E_wall_erg += dE_wall_hJ * 1e9
                E_wall_array_erg[i] = E_wall_erg

            F[i] = base_flux - (dE_wall_hJ * 1e9) / (A[i] * dt_i)

            # your ablation E update included A_n/area factor
            E[i] = E[i - 1] + 0.5 * (F[i] * A[i] + F[i - 1] * A[i-1]) * dt_i / area

        elif wall_loss:
            dE_wall_hJ = WallLossModel.compute_wall_energy_loss(t_sec[i], dt_i, t_heat, R_cm, Ts_prev, xF[i - 1], flat_top_profile=True, wall=wall_material)

            E_wall_erg += dE_wall_hJ * 1e9
            E_wall_array_erg[i] = E_wall_erg

            F[i] = base_flux - (dE_wall_hJ * 1e9) / (area * dt_i)
            E[i] = E[i - 1] + 0.5 * (F[i] + F[i - 1]) * dt_i

        else:
            E_wall_array_erg[i] = E_wall_erg
            F[i] = base_flux
            E[i] = E[i - 1] + 0.5 * (F[i] + F[i - 1]) * dt_i

        # ---- If vary_rho: update rho_eff and C_changing_rho and Z1 ----
        if vary_rho and ablation:
            new_rho[i] = AblationModel.compute_rho_effective(R_cm, R_array, xF[i - 1])
            if not lam_eff:
                In[i] = In[i-1] + dt_i/2 * (new_rho[i]**p + new_rho[i-1]**p)
                C_changing_rho[i] = (K / t_sec[i]) * In[i]
                Z1_changing_rho[i] = (f ** 2) * (2.0 + eps) * (1.0 - eps) * C_changing_rho[i] * (new_rho[i] ** (2.0 * (1.0 - mu)))
            else:
                lambda_ross = g*(Ts_prev**alpha)*(rho**(-lambda_param-1))
                xF_index = np.searchsorted(z, xF[i-1])
                if R_average_for_lambda_geom:
                    R_average = np.mean(R_array[:xF_index+1]) if xF_index> 0 else R_array[0]
                    #R_average = R_array[0]
                else:
                    R_average = R_cm
                lambda_geom = 2*R_average
                lambda_eff = ((lambda_geom**(-power) + lambda_ross**(-power)))**(-1/power)
                lambda_eff_array[i] = lambda_eff
                g_eff = g * ( lambda_eff / lambda_ross )
                g_eff_array[i] = g_eff

                    # --- integrate (g_eff/g)*rho^p together ---
                w_i   = g_eff_array[i]   / g
                w_im1 = g_eff_array[i-1] / g  # make sure g_eff_array[0] is initialized!

                In[i] = In[i-1] + 0.5 * dt_i * (
                    w_i   * (new_rho[i]   ** p) +
                    w_im1 * (new_rho[i-1] ** p)
                )
                # --- now C uses the combined integral directly ---
                C_changing_rho[i] = (K / t_sec[i]) * In[i]

                C_eff_array[i] = C_changing_rho[i]  # if you don't need separate C_eff
                Z1_changing_rho[i] = (f**2) * (2.0 + eps) * (1.0 - eps) * C_changing_rho[i] * (new_rho[i] ** (2.0 * (1.0 - mu)))
        
            
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

        # update t_heat
        for j in range(len(z)):
            if z[j] <= xF[i] and t_heat[j] == np.inf:
                t_heat[j] = t_sec[i]
        
        if wall_loss and i > 1:
            dE_wall = E_wall_array_erg[i] - E_wall_array_erg[i - 1]
            albedo = AlbedoModel.compute_albedo_step(Ts[i], dE_wall, dt_i)
            lambda_ross = g*(Ts[i]**alpha)*(rho**(-lambda_param-1))
            epsilon = 3/4*(1 - albedo)*(1/lambda_ross)*R_cm
            kappa_0 = kappa_roots(epsilon, R_cm, n_roots=1)[0]
            kappa_0_approx = np.sqrt(2*epsilon)/R_cm  # for small epsilon, J_0(kappa_0*R_cm) ~ 0 => kappa_0 ~ sqrt(2*epsilon/R_cm)
            
            # Compute J_0(kappa_0 * r) for each z position
            J0_profiles = np.zeros((len(z), len(r_grid)))
            J0_profiles_approx = np.zeros((len(z), len(r_grid)))
            for j_idx in range(len(z)):
                # J_0(kappa_0 * r / R_cm) for each radial point
                J0_profiles[j_idx, :] = special.j0(kappa_0 * r_grid)
                J0_profiles_approx[j_idx, :] = special.j0(kappa_0_approx * r_grid)
            
            # Store data with time as key (in nanoseconds)
            t_ns = t_sec[i] * 1e9
            bessel_data[t_ns] = {
                'r_grid': r_grid.copy(),  # radial grid in cm
                'z_grid': z.copy(),        # axial grid in cm
                'J0_profiles': J0_profiles.copy(),  # shape: (Nz, Nr), J_0(kappa_0*r) at each (z, r)
                'J0_profiles_approx': J0_profiles_approx.copy(),  # shape: (Nz, Nr), J_0(kappa_0_approx*r) at each (z, r)
                'kappa_0': kappa_0,
                'kappa_0_approx': kappa_0_approx,
                'epsilon': epsilon,
                'albedo': albedo,
                'lambda_ross': lambda_ross
            }
            
    # Convert energies to hJ like your return:
    # You returned: E*1e-9*area, E_wall_array*1e-9
    # That's (erg/cm^2)*area -> erg ; *1e-9 -> hJ (since 1 hJ = 1e-7 J = 1e0 erg? careful)
    # I will keep EXACTLY your conversion factors to preserve your downstream plots.
    E_total_hJ = E * 1e-9 * area
    E_wall_hJ_array = E_wall_array_erg * 1e-9

    # restore original order for all arrays
    xF_out = WavefrontHelpers.restore_original_order(xF, order, t_sec_in.size) / 1.02
    Ts_out = WavefrontHelpers.restore_original_order(Ts, order, t_sec_in.size)
    E_out  = WavefrontHelpers.restore_original_order(E_total_hJ, order, t_sec_in.size)
    Ew_out = WavefrontHelpers.restore_original_order(E_wall_hJ_array, order, t_sec_in.size)
    #truen the times in data_of_R to ns
    data_of_R = {1e9*t_sec_in[i]: data_of_R[t_sec[i]] for i in range(t_sec_in.size)}
    # if vary_rho and ablation:
    #     plt.plot(t_sec*1e9, new_rho)
    #     plt.xlabel("Time (ns)")
    #     plt.ylabel("Effective Density (g/cm^3)")
    #     plt.title("Effective Density vs Time")
    #     plt.grid()
    #     plt.show()
    # power = 10
    # lambda_ross = g*(Ts**alpha)*(rho**(-lambda_param-1))
    # lambda_geom = 2*R_cm
    # lambda_param_eff_array = [lambda_ross_i * lambda_geom * (1/(lambda_geom**power + lambda_ross_i**power))**(1/power) for lambda_ross_i in lambda_ross]
    # if lam_eff:
    #     lambda_ross_array= g*(Ts**alpha)*(rho**(-lambda_param-1))
    #     plt.plot(t_sec*1e9, lambda_eff_array, label="Effective lambda_param")
    #     plt.plot(t_sec*1e9, lambda_ross_array, label="lambda ross")
    #     plt.plot(t_sec*1e9, [lambda_geom for _ in t_sec], label="Geometric lambda_param")
    #     plt.xlabel("Time (ns)")
    #     plt.ylabel("g_eff / g")
    #     plt.xlim(0.01, t_sec[-1]*1e9)
    #     plt.ylim(0, 0.13)
    #     plt.title(f"Effective g vs Time {{wall_loss = {wall_loss}, ablation={ablation}, vary_rho={vary_rho}}}")
    #     plt.grid()
    #     plt.legend()
    #     plt.show()
    return xF_out, Ts_out, E_out, Ew_out, data_of_R, bessel_data


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

def analytic_wave_front_marshak_ablation(times_to_store, *, use_seconds=True, vary_rho=False, wall_material='Gold', lam_eff=False, power=2, R_average_for_lambda_geom=False, good_way=False):
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
        good_way=good_way,
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
    R_average_for_lambda_geom=False,
    good_way=True,
):
    """
    mode options:
      - "no_marshak"
      - "marshak"
      - "marshak_wall_loss"
      - "marshak_ablation"
    """
    return _get_default_solver().analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=use_seconds,
        mode=mode,
        vary_rho=vary_rho,
        wall_material=wall_material,
        lam_eff=lam_eff,
        power=power,
        R_average_for_lambda_geom=R_average_for_lambda_geom,
        good_way=good_way,
    )

######################################################################################

_DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER = None


def _get_default_solver():
    global _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER
    if _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER is None:
        _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER = AnalyticalWavefrontSolver(
            no_marshak_fn=analytic_wave_front_no_marshak,
            march_fn=_marshak_appendixA_march,
        )
    return _DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER
