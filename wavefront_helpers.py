import numpy as np
import scipy
from parameters import *


class WavefrontHelpers:
    """Utility helpers used by analytical wave-front models."""

    @staticmethod
    def compute_constants_for_wavefront():
        """
        Uses globals you already rely on:
        alpha, beta, f, g, mu, lambda_param, rho, a_hev
        Returns: eps, sigma_SB_hev, C, pref
        """
        eps = beta / (4.0 + alpha)

        c_cgs = 3.0e10
        sigma_SB_hev = a_hev * c_cgs / 4.0

        C = (16.0 / (4.0 + alpha)) * (g * sigma_SB_hev) / (3.0 * f * rho ** (2.0 - mu + lambda_param))
        pref = (2.0 + eps) / (1.0 - eps)
        return eps, sigma_SB_hev, C, pref

    @staticmethod
    def solve_for_H_new_brentq(Z1, eps, E2, I_prev, H_prev, dt):
        """
        Implicit Eq. (A.3), same as your current solve_for_H_new, but fully parameterized.
        Solves for H_new > 0:
            Z1 * ( I_prev + 0.5*(H_prev + H_new)*dt ) * H_new^eps - E2 = 0
        """
        def fH(Hx):
            return Z1 * (I_prev + 0.5 * (H_prev + Hx) * dt) * (Hx ** eps) - E2

        H_new = scipy.optimize.brentq(fH, 1e-100, 1e50, maxiter=100)
        return H_new

    @staticmethod
    def prepare_times(times_to_store, use_seconds=True):
        times_to_store = np.asarray(times_to_store, dtype=float)
        if times_to_store.size == 0:
            return times_to_store, None, None
        t_sec_in = times_to_store if use_seconds else times_to_store * 1e-9
        order = np.argsort(t_sec_in)
        t_sec = t_sec_in[order]
        if t_sec[-1] > 1e-5:
            t_sec = t_sec * 1e-9
            t_sec_in = t_sec_in * 1e-9
        return t_sec, order, t_sec_in

    @staticmethod
    def restore_original_order(arr, order, original_size):
        """Restore results to original time order after sorted computation."""
        out = np.empty(original_size, dtype=float)
        out[order] = arr
        return out

    @staticmethod
    def compute_first_order_hr_profile(t_sec, xF_arr, Ts_arr, z_grid=None, C_val=None):
        """
        Computes the first-order corrected Hammer-Rosen temperature profile T(z, t)
        and the profile parameter A(t) (Appendix A / Eqs. 22-23 of HR_corrected.tex).
        """
        eps, _, C_default, _ = WavefrontHelpers.compute_constants_for_wavefront()
        if C_val is None:
            C_val = C_default
        if z_grid is None:
            z_grid = z

        N = len(t_sec)
        N_x = len(z_grid)
        n = 4.0 + alpha

        H_arr = Ts_arr ** n
        Hdot = np.zeros(N)
        for i in range(1, N):
            dt_i = t_sec[i] - t_sec[i - 1]
            if dt_i > 0:
                Hdot[i] = (H_arr[i] - H_arr[i - 1]) / dt_i
        if N > 1 and (t_sec[1] - t_sec[0]) > 0:
            Hdot[0] = (H_arr[1] - H_arr[0]) / (t_sec[1] - t_sec[0])

        A_arr = np.zeros(N)
        for i in range(N):
            if H_arr[i] > 1e-100 and xF_arr[i] > 0 and C_val > 0:
                H_1meps = H_arr[i] ** (1.0 - eps)
                if H_1meps > 1e-100:
                    A_arr[i] = (xF_arr[i] ** 2) / (C_val * H_1meps) * Hdot[i] / H_arr[i]

        T_profile_leading = np.zeros((N, N_x))
        T_profile_corrected = np.zeros((N, N_x))
        for i in range(N):
            if xF_arr[i] < 1e-30 or Ts_arr[i] < 1e-30:
                continue
            y = z_grid / xF_arr[i]
            for j in range(N_x):
                if y[j] < 1.0:
                    T_profile_leading[i, j] = Ts_arr[i] * ((1.0 - y[j]) ** (1.0 / n))
                    inner = (1.0 - y[j]) * (1.0 + (eps / 2.0) * (1.0 - A_arr[i]) * y[j])
                    if inner > 0:
                        T_profile_corrected[i, j] = Ts_arr[i] * (inner ** (1.0 / (n * (1.0 - eps))))
        return A_arr, T_profile_leading, T_profile_corrected

    @staticmethod
    def compute_optical_depth_and_mfp(xF, Ts, cutoff_fraction=0.9, use_user_formula=True):
        r"""
        Computes the optical depth (`tau`) and mean free path (`mean_free_path`) across
        the spatial region [0, cutoff_fraction * xF] behind the wavefront using the
        analytical Henyey self-similar temperature profile.

        Parameters:
        -----------
        xF : float or np.ndarray
            Wavefront penetration depth [cm]
        Ts : float or np.ndarray
            Surface temperature [HeV]
        cutoff_fraction : float
            Fraction of the wavefront to integrate over (default: 0.9)
        use_user_formula : bool
            If True (default):
                tau = \int_0^{cutoff_fraction * xF} (1 / (kappa_R * rho)) dx
                mean_free_path = (cutoff_fraction * xF) * tau
            If False (standard physical definition):
                tau = \int_0^{cutoff_fraction * xF} (kappa_R * rho) dx
                mean_free_path = (cutoff_fraction * xF) / tau
        
        Returns:
        --------
        tau : float or np.ndarray
        mean_free_path : float or np.ndarray
        """
        xF_arr = np.atleast_1d(np.asarray(xF, dtype=float))
        Ts_arr = np.atleast_1d(np.asarray(Ts, dtype=float))

        tau_out = np.zeros_like(xF_arr)
        mfp_out = np.zeros_like(xF_arr)

        mask = (xF_arr > 1e-30) & (Ts_arr > 1e-30)
        if not np.any(mask):
            if np.isscalar(xF) and np.isscalar(Ts):
                return 0.0, 0.0
            return tau_out, mfp_out

        # Exponent for Henyey profile: T(u) = Ts * (1 - u)^(1 / (4 + alpha - beta))
        # Rosseland mean free path: l_R(u) = g * T(u)^alpha * rho^(-lambda_param - 1)
        #                           = l_0 * (1 - u)^gamma
        gamma = alpha / (4.0 + alpha - beta)
        l_0 = g * (Ts_arr[mask] ** alpha) * (rho ** (-lambda_param - 1.0))

        # Physical optical depth: tau = int_0^{u_c * xF} (1 / l_R) dx = (xF / l_0) * [1 - (1 - u_c)^(1 - gamma)] / (1 - gamma)
        tau_mask = (xF_arr[mask] / l_0) * (1.0 - (1.0 - cutoff_fraction) ** (1.0 - gamma)) / (1.0 - gamma)
        tau_out[mask] = tau_mask
        mfp_out[mask] = (cutoff_fraction * xF_arr[mask]) / np.maximum(tau_mask, 1e-100)

        if np.isscalar(xF) and np.isscalar(Ts):
            return float(tau_out[0]), float(mfp_out[0])
        return tau_out, mfp_out
