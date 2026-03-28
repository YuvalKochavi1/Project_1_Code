import numpy as np
from parameters import *


class AlbedoModel:
    """Albedo calculation model."""

    @staticmethod
    def compute_albedo_step(T_s_hev, dE_wall_erg, dt_sec):
        """
        Compute albedo at a single time step from the flux balance.

        The albedo is defined as the ratio of reflected to incident flux:
            a(t) = F_out / F_in = (σ_SB * T_s^4 * area - Ė_w/2) / (σ_SB * T_s^4 * area + Ė_w/2)

        where:
          - T_s: surface temperature in HeV
          - Ė_w = dE_wall / dt: rate of energy loss to wall (erg/s)
          - σ_SB: Stefan–Boltzmann constant

        Parameters
        ----------
        T_s_hev : float
            Surface temperature in HeV
        dE_wall_erg : float
            Energy loss to wall in this time step (erg)
        dt_sec : float
            Time step duration (seconds)

        Returns
        -------
        float
            Albedo value (dimensionless)
        """
        if T_s_hev <= 0 or dt_sec <= 0:
            return 0.0

        sigma_SB_hev = a_hev * 3e10 / 4.0
        wall_flux_rate = dE_wall_erg / dt_sec
        F_in = sigma_SB_hev * (T_s_hev ** 4.0) * np.pi * R_cm**2 + 0.5 * wall_flux_rate
        # print(f"wall_flux_rate={wall_flux_rate/2}, sigma*Ts**4*pi*R**2={sigma_SB_hev * (T_s_hev ** 4.0)*np.pi*R_cm**2}")

        F_out = sigma_SB_hev * (T_s_hev ** 4.0) * np.pi * R_cm**2 - 0.5 * wall_flux_rate
        if F_out <= 0:
            return np.inf
        albedo = F_out / F_in
        return albedo

    @staticmethod
    def compute_albedo(t_array, T_s_array, E_wall_array_erg):
        """
        Compute albedo array from stored time series data.

        Albedo is computed at each time step using the flux balance equation:
            a(t) = F_out / F_in = (σ_SB * T_s^4 * area - Ė_w/2) / (σ_SB * T_s^4 * area + Ė_w/2)

        Parameters
        ----------
        t_array : np.ndarray
            Time array (seconds), shape (Nt,)
        T_s_array : np.ndarray
            Surface (Marshak) temperature array (HeV), shape (Nt,)
        E_wall_array_erg : np.ndarray
            Cumulative wall energy loss (erg), shape (Nt,)

        Returns
        -------
        np.ndarray
            Albedo as a function of time, shape (Nt,), computed at each step
        """
        t_array = np.asarray(t_array, dtype=float)
        T_s_array = np.asarray(T_s_array, dtype=float)
        E_wall_array_erg = np.asarray(E_wall_array_erg, dtype=float)

        if t_array.size == 0:
            return np.array([])

        albedo = np.zeros_like(t_array)
        albedo[0] = 0.0

        for i in range(1, len(t_array)):
            dt = t_array[i] - t_array[i - 1]
            dt_sec = dt / 1e9
            dE_wall = E_wall_array_erg[i] - E_wall_array_erg[i - 1]
            albedo[i] = AlbedoModel.compute_albedo_step(T_s_array[i], dE_wall, dt_sec)

        return albedo
