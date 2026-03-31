import numpy as np
from parameters import *
from wall_loss_model import WallLossModel


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

    @staticmethod
    def compute_wall_energy_loss_profile_at_interface(
        t_sec, dt_sec, t_heat, T_profile_z, xF, wall_material='Gold', R_ablation_profile=None
    ):
        """
        Compute wall energy loss per z-cell at the foam-gold interface.
        
        Evaluates wall areal loading (and hence penetration depth) at each z-cell
        at either r=R_cm (no ablation) or r=R_ablation(z) (with ablation),
        then returns both the per-cell profile and the spatially-averaged value.

        Parameters
        ----------
        t_sec : float
            Current time in seconds.
        dt_sec : float
            Timestep in seconds.
        t_heat : np.ndarray
            Heating time for each z-cell (seconds), shape (Nz,).
        T_profile_z : np.ndarray or float
            Temperature profile or constant value at interface.
            If array, shape (Nz,).
            If float, constant T_0 everywhere.
        xF : float
            Heat front position at r=R_cm (cm).
        wall_material : str
            Wall material ('Gold', 'Copper', 'Be').
        R_ablation_profile : np.ndarray, optional
            Ablation radius profile R(z), shape (Nz,).
            If None, uses fixed R_cm everywhere (non-ablation case).

        Returns
        -------
        tuple
            (dE_wall_profile_per_cell, dE_wall_averaged)
            - dE_wall_profile_per_cell: incremental penetration depth per z-cell (cm)
            - dE_wall_averaged: spatially-averaged penetration depth (cm)
        """
        z_array = np.asarray(z, dtype=float)
        t_heat = np.asarray(t_heat, dtype=float)
        
        # Convert scalar temperature to profile if needed
        if np.isscalar(T_profile_z):
            T_profile = np.full_like(z_array, T_profile_z, dtype=float)
        else:
            T_profile = np.asarray(T_profile_z, dtype=float)
        
        # Use R_cm if no ablation profile provided
        if R_ablation_profile is None:
            R_ablation_profile = np.full_like(z_array, R_cm, dtype=float)
        else:
            R_ablation_profile = np.asarray(R_ablation_profile, dtype=float)
        
        # Compute incremental areal loading for each z-cell at the interface
        dE_wall_profile = np.zeros_like(z_array, dtype=float)
        
        for i in range(len(z_array)):
            if z_array[i] > xF:
                # Cell beyond front: no energy loss
                dE_wall_profile[i] = 0.0
                continue
            
            t_exposed = t_sec - t_heat[i]
            if t_exposed <= 0:
                dE_wall_profile[i] = 0.0
                continue
            
            T_local = float(T_profile[i])
            if not np.isfinite(T_local) or T_local <= 0:
                dE_wall_profile[i] = 0.0
                continue
            
            # Get material-specific areal loading function
            if wall_material == 'Gold':
                areal_loading_fn = WallLossModel.gold_areal_loading_g_per_cm2
                rho_wall = rho_gold
            elif wall_material == 'Copper':
                areal_loading_fn = WallLossModel.Copper_areal_loading_g_per_cm2
                rho_wall = rho_copper
            elif wall_material == 'Be':
                areal_loading_fn = WallLossModel.Be_areal_loading_g_per_cm2
                rho_wall = rho_be
            else:
                dE_wall_profile[i] = 0.0
                continue
            
            # Cumulative areal loading approach (not incremental)
            m_now = areal_loading_fn(max(t_exposed, 0.0), T_local)
            dE_wall_profile[i] = m_now / rho_wall  # convert to penetration depth
        
        # Compute spatial average (only over cells within front)
        within_front = z_array <= xF
        if np.any(within_front):
            dE_wall_averaged = np.mean(dE_wall_profile[within_front])
        else:
            dE_wall_averaged = 0.0
        
        return dE_wall_profile, dE_wall_averaged
