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
            Wall material ('Gold', 'Copper', 'Be', 'Vacuum', ...).

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
                    elif wall == "Copper":
                        delta_e_i = WallLossModel.E_wall_copper(t_exposed, T0) - WallLossModel.E_wall_copper(t_exposed - dt, T0)
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
                    elif wall == 'Copper':
                        xi = z[i]
                        if xi < xF and xF > 0:
                            exponent = 1.0 / (4.0 + alpha - beta)
                            T_local = T0 * (1 - xi / xF) ** exponent
                            delta_e_i = WallLossModel.E_wall_copper(t_exposed, T_local) - WallLossModel.E_wall_copper(t_exposed - dt, T_local)
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
    def E_wall_array_dt(t, dt, t_heat, T0, xF, flat_top_profile=True, wall='Gold'):
        """Compute wall energy loss per z-cell at the foam-gold interface for an array of z values.
        Will be used to compute the wall albedo profile as a function of z at the interface."""
        dE_array = np.zeros_like(z, dtype=float)
        for i in range(len(z)):
            if z[i] <= xF:
                t_exposed = t - t_heat[i]
                if flat_top_profile:
                    if wall == 'Gold':
                        delta_e_i = WallLossModel.E_wall_gold(t_exposed, T0) - WallLossModel.E_wall_gold(t_exposed - dt, T0)
                    elif wall == "Copper":
                        delta_e_i = WallLossModel.E_wall_copper(t_exposed, T0) - WallLossModel.E_wall_copper(t_exposed - dt, T0)
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
                    elif wall == 'Copper':
                        xi = z[i]
                        if xi < xF and xF > 0:
                            exponent = 1.0 / (4.0 + alpha - beta)
                            T_local = T0 * (1 - xi / xF) ** exponent
                            delta_e_i = WallLossModel.E_wall_copper(t_exposed, T_local) - WallLossModel.E_wall_copper(t_exposed - dt, T_local)
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
                dE_array[i] = delta_e_i
        dE_array_erg = dE_array * 1e11
        return dE_array_erg

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
        #gold subsonic
        return 0.59 * T0**3.35 * (t_exposed * 1e9)**0.59
        #gold supersonic
        g0gold = 1/7200
        return 0.175 * ((g_gold/g0gold) ** 0.5) * (T0 ** 3.55) * ((t_exposed * 1e9) ** 0.5) 

    @staticmethod
    def E_wall_copper(t_exposed, T0):
        """Wall energy loss for copper in hJ/mm^2 (Eq. 11 parameterization)."""
        if t_exposed <= 0:
            return 0.0
        return 1.58 * T0**3.4 * (t_exposed * 1e9)**0.61
        #return 0.4 * T0**3.78 * (t_exposed * 1e9)**0.5

    @staticmethod
    def E_wall_be(t_exposed, T0):
        """Wall energy loss for beryllium in hJ/mm^2 (Eq. 11 parameterization)."""
        if t_exposed <= 0:
            return 0.0
        g0be = 1/402.8
        return 1.27 * ((g_be/g0be) ** 0.5) * T0**4.99 * (t_exposed * 1e9) ** 0.5
    
    @staticmethod
    def gold_areal_loading_g_per_cm2(t_exposed, T0):
        """Gold areal loading sigma_Au = rho_Au * zF_Au in g/cm^2."""
        if t_exposed <= 0:
            return 0.0
        return 10.17e-4 * (T0 ** 1.91) * ((t_exposed * 1e9) ** 0.52)
    
    @staticmethod
    def Copper_areal_loading_g_per_cm2(t_exposed, T0):
        """Copper areal loading sigma_Cu = rho_Cu * zF_Cu in g/cm^2."""
        if t_exposed <= 0:
            return 0.0
        return 1.2e-3 * (T0 ** 2.43) * ((t_exposed * 1e9) ** 0.5)
    
    @staticmethod
    def Be_areal_loading_g_per_cm2(t_exposed, T0):
        """Beryllium areal loading sigma_Be = rho_Be * zF_Be in g/cm^2."""
        if t_exposed <= 0:
            return 0.0
        return 1.71e-3 * (T0 ** 3.9) * ((t_exposed * 1e9) ** 0.5)

    @staticmethod
    def wall_penetration_depth_cm(t_exposed, T0, wall_material='Gold'):
        """Penetration depth for the specified wall material in cm from sigma_Au = rho_Au * zF_Au."""
        if wall_material == 'Gold':
            sigma_au = WallLossModel.gold_areal_loading_g_per_cm2(t_exposed, T0)
            rho_wall = rho_gold
        elif wall_material == 'Copper':
            sigma_au = WallLossModel.Copper_areal_loading_g_per_cm2(t_exposed, T0)
            rho_wall = rho_copper
        elif wall_material == 'Be':
            sigma_au = WallLossModel.Be_areal_loading_g_per_cm2(t_exposed, T0)
            rho_wall = rho_be
        else:
            sigma_au = 0.0
        return sigma_au / rho_wall

    @staticmethod
    def shock_packing_pressure(T0):
        """Packing pressure P0 as a function of interface temperature T0."""
        return 2.71 * (T0 ** 2.63)

    @staticmethod
    def shock_mass_parameter(t_exposed, T0):
        """Shock mass parameter m_s using the local exposure time in seconds."""
        if t_exposed <= 0:
            return 0.0
        P0 = WallLossModel.shock_packing_pressure(T0)
        return 5.38e-3 * (P0 ** 0.5) * t_exposed

    @staticmethod
    def shock_penetration_depth_cm(t_exposed, T0):
        """Approximate shock-front penetration depth in cm."""
        return WallLossModel.shock_mass_parameter((t_exposed * 1e9), T0) / rho_gold

    @staticmethod
    def compute_shock_front_profile(t, dt, t_heat, T0, xF, *, wall='Gold'):
        """Compute the shock-front penetration depth profile on the global z-grid."""
        shock_profile = np.zeros_like(z, dtype=float)
        if wall != 'Gold':
            return shock_profile

        for i in range(len(z)):
            if z[i] <= xF:
                t_exposed = t - t_heat[i]
                if t_exposed <= 0:
                    continue
                T_local = T0 # it's flattop since its hydrodynamics and not marshak waves
                shock_profile[i] = max(WallLossModel.shock_penetration_depth_cm(t_exposed, T_local), 0.0)
        return shock_profile

    @staticmethod
    def compute_wall_front_profile(t, dt, t_heat, T0, xF, *, flat_top_profile=True, wall='Gold'):
        """
        Compute wall areal loading profile sigma_{wall}(z) = rho_{wall} * zF_{wall}(z).

        If incremental=True, returns delta sigma_{wall} over the current timestep dt.
        If incremental=False, returns cumulative sigma_{wall} at time t.

        Parameters
        ----------
        t : float
            Current time in seconds.
        dt : float
            Timestep in seconds.
        t_heat : np.ndarray
            Heating time for each z-cell in seconds.
        T0 : float
            Surface temperature in HeV.
        xF : float
            Front position in cm.
        flat_top_profile : bool
            If True use T0 everywhere behind the front; otherwise use self-similar T(z).
        wall : str
            Wall material. Currently supports 'Gold'.
        incremental : bool
            Whether to return delta profile (True) or cumulative profile (False).

        Returns
        -------
        np.ndarray
            Areal loading profile in g/cm^2 on the global z-grid.
        """
        m_profile = np.zeros_like(z, dtype=float)
        if wall == 'Gold':
            exponent = 1.0 / (4.0 + alpha_gold - beta_gold)
            rho_wall = rho_gold
        elif wall == 'Copper':
            exponent = 1.0 / (4.0 + alpha_copper - beta_copper)
            rho_wall = rho_copper
        elif wall == 'Be':
            exponent = 1.0 / (4.0 + alpha_be - beta_be)
            rho_wall = rho_be
        else: 
            return np.zeros_like(z, dtype=float)

        for i in range(len(z)):
            if z[i] <= xF:
                t_exposed = t - t_heat[i]
                if t_exposed <= 0:
                    continue
                if flat_top_profile:
                    T_local = T0
                else:
                    if xF <= 0 or z[i] >= xF:
                        T_local = 0.0
                    else:
                        T_local = T0 * (1.0 - z[i] / xF) ** exponent

                m_now = WallLossModel.gold_areal_loading_g_per_cm2(t_exposed, T_local)
                m_prev = WallLossModel.gold_areal_loading_g_per_cm2(t_exposed - dt, T_local)
                m_profile[i] = max(m_now - m_prev, 0.0)
        return m_profile / rho_wall
