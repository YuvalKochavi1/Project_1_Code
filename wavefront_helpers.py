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
