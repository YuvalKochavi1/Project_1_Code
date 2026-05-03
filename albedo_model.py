import numpy as np
from parameters import *
from wall_loss_model import WallLossModel

dz = z[1] - z[0]

class AlbedoModel:
    """Albedo calculation model."""

    @staticmethod
    def compute_albedo_step(T_s_hev, dE_wall_erg, dt_sec, x_F):
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
        # Use the same interface area as compute_wall_energy_loss: 2*pi*R_cm*x_F
        area = 2.0 * np.pi * R_cm * x_F
        F_in = sigma_SB_hev * (T_s_hev ** 4.0) * area + 0.5 * wall_flux_rate
        F_out = sigma_SB_hev * (T_s_hev ** 4.0) * area - 0.5 * wall_flux_rate
        if F_out <= 0:
            return np.clip(0.0, 0.0, 1.0 - 1e-12)
        albedo = F_out / F_in
        albedo = float(np.clip(albedo, 0.0, 1.0 - 1e-12))
        return albedo


    @staticmethod
    def compute_albedo_profile(
        t_sec, dt_sec, t_heat, T_s, xF, wall_material='Gold', R_ablation_profile=None
    ):
        if T_s <= 0 or dt_sec <= 0:
            return 0.0

        dE_wall_array = WallLossModel.E_wall_array_dt(t_sec, dt_sec, t_heat, T_s, xF, flat_top_profile=True, wall='Gold')
        sigma_SB_hev = a_hev * 3e10 / 4.0
        wall_flux_rate = dE_wall_array / dt_sec
        T_s_array = np.full_like(wall_flux_rate, T_s)
        F_in = sigma_SB_hev * (T_s_array ** 4.0) + 0.5 * wall_flux_rate
        F_out = sigma_SB_hev * (T_s_array ** 4.0) - 0.5 * wall_flux_rate
        # if F_out <= 0:
        #     return np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            albedo = F_out / F_in
        albedo = np.asarray(albedo, dtype=float)
        albedo = np.nan_to_num(albedo, nan=0.0, posinf=1.0, neginf=0.0)

        # find x_front index and keep averaging robust when the front is near z=0
        x_front_index = int(np.searchsorted(z, xF))
        if x_front_index <= 0:
            avg_albedo = float(albedo[0]) if albedo.size > 0 else 0.0
        else:
            avg_albedo = float(np.mean(albedo[:x_front_index]))

        avg_albedo = float(np.clip(np.nan_to_num(avg_albedo, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
        return albedo, avg_albedo
