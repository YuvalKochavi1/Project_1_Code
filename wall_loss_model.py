import numpy as np
from parameters import *


class WallLossModel:
    """Wall-loss physics model."""

    @staticmethod
    def compute_wall_energy_loss(t, dt, t_heat, R_cm_local, T0, xF, flat_top_profile=True, wall='Gold'):
        """
        Compute energy loss to the wall at time t (sec) using Eq. (10) and (11) of article 2.

        Parameters
        ----------
        t : float
            Current time in seconds.
        t_heat : np.ndarray
            Array of heating times for each spatial point in seconds.
        R_cm_local : float
            Radius of the cylinder (foam part) in cm.
        T0 : float
            Surface temperature in HeV.
        wall : str
            Wall material ('Gold', 'Cupper', 'Be', 'Vacuum', ...).

        Returns
        -------
        float
            Total energy loss to the wall in hJ at time t.
        """
        dE_total = 0.0
        for i in range(len(z)):
            if z[i] <= xF:
                t_exposed = t - t_heat[i]
                if flat_top_profile:
                    if wall == 'Gold':
                        delta_e_i = WallLossModel.E_wall_gold(t_exposed, T0) - WallLossModel.E_wall_gold(t_exposed - dt, T0)
                    elif wall == "Cupper":
                        delta_e_i = WallLossModel.E_wall_gold(t_exposed, T0) - WallLossModel.E_wall_gold(t_exposed - dt, T0)
                    elif wall == 'Be':
                        delta_e_i = WallLossModel.E_wall_be(t_exposed, T0) - WallLossModel.E_wall_be(t_exposed - dt, T0)
                    elif wall == 'Vacuum':
                        delta_e_i = WallLossModel.delta_e_vacuum_hJ_per_mm2(dt, T0)
                    else:
                        delta_e_i = 0.0
                else:
                    if wall == 'Gold':
                        xi = z[i]
                        if xi < xF and xF > 0:
                            exponent = 1.0 / (4.0 + alpha - beta)
                            T_local = T0 * (1 - xi / xF) ** exponent
                            delta_e_i = WallLossModel.E_wall_gold(t_exposed, T_local) - WallLossModel.E_wall_gold(t_exposed - dt, T_local)
                        else:
                            delta_e_i = 0.0
                    elif wall == 'Be':
                        xi = z[i]
                        if xi < xF and xF > 0:
                            exponent = 1.0 / (4.0 + alpha - beta)
                            T_local = T0 * (1 - xi / xF) ** exponent
                            delta_e_i = WallLossModel.E_wall_be(t_exposed, T_local) - WallLossModel.E_wall_be(t_exposed - dt, T_local)
                        else:
                            delta_e_i = 0.0
                    elif wall == 'Cupper':
                        xi = z[i]
                        if xi < xF and xF > 0:
                            exponent = 1.0 / (4.0 + alpha - beta)
                            T_local = T0 * (1 - xi / xF) ** exponent
                            delta_e_i = WallLossModel.E_wall_cupper(t_exposed, T_local) - WallLossModel.E_wall_cupper(t_exposed - dt, T_local)
                        else:
                            delta_e_i = 0.0
                    elif wall == 'Vacuum':
                        xi = z[i]
                        if xi < xF and xF > 0:
                            exponent = 1.0 / (4.0 + alpha - beta)
                            T_local = T0 * (1 - xi / xF) ** exponent
                            delta_e_i = WallLossModel.delta_e_vacuum_hJ_per_mm2(dt, T_local)
                        else:
                            delta_e_i = 0.0
                    else:
                        delta_e_i = 0.0
                dE_total += delta_e_i * dz * 10
        return 2 * np.pi * R_cm_local * 10 * dE_total

    @staticmethod
    def delta_e_vacuum_hJ_per_mm2(dt_sec, T_hev):
        """Vacuum radiative energy loss increment in hJ/mm^2 for a timestep."""
        sigma_SB_cgs = 5.670374419e-5
        T_K = T_hev * K_per_Hev
        dE_erg_per_cm2 = sigma_SB_cgs * (T_K**4) * dt_sec
        dE_hJ_per_mm2 = dE_erg_per_cm2 * (1e-9) / 100.0
        return dE_hJ_per_mm2

    @staticmethod
    def E_wall_gold(t_exposed, T0):
        """Wall energy loss for gold in hJ/mm^2 (Eq. 11 parameterization)."""
        if t_exposed <= 0:
            return 0.0
        return 0.59 * T0**3.35 * (t_exposed * 1e9)**0.59

    @staticmethod
    def E_wall_gold_dot(t_exposed, T0):
        """Time-derivative form for gold wall loss in hJ/(mm^2*s)."""
        if t_exposed <= 0:
            return 0.0
        return (0.59)**2 * T0**3.35 * (t_exposed * 1e9)**(-0.41)

    @staticmethod
    def E_wall_cupper(t_exposed, T0):
        """Wall energy loss for copper in hJ/mm^2 (Eq. 11 parameterization)."""
        if t_exposed <= 0:
            return 0.0
        return 1.58 * T0**3.4 * (t_exposed * 1e9)**0.61

    @staticmethod
    def E_wall_be(t_exposed, T0):
        """Wall energy loss for beryllium in hJ/mm^2 (Eq. 11 parameterization)."""
        if t_exposed <= 0:
            return 0.0
        return 1.27 * T0**4.99 * (t_exposed * 1e9)**0.5
