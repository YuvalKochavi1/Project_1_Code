import numpy as np
import pandas as pd
from parameters import *


class AblationModel:
    """Ablation and effective-density model."""

    @staticmethod
    def get_u_tilda_closest(csv_path: str, rho_query: float) -> float:
        """
        Read CSV [rho, u_tilda] and return nearest-neighbor u_tilda for rho_query.
        """
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least two columns: rho, u_tilda")

        rho_arr = df.iloc[1:, 0].to_numpy(dtype=float)
        u_arr = df.iloc[1:, 1].to_numpy(dtype=float)

        order_1 = np.argsort(rho_arr)
        rho_arr = rho_arr[order_1]
        u_arr = u_arr[order_1]
        rho_query = 4 * float(rho_query)

        i = np.searchsorted(rho_arr, rho_query)

        if i == 0:
            return u_arr[0]
        if i >= len(rho_arr):
            return u_arr[-1]

        left = i - 1
        right = i
        if abs(rho_arr[right] - rho_query) < abs(rho_arr[left] - rho_query):
            return u_arr[right]
        return u_arr[left]

    @staticmethod
    def ablation_velocity_gold(t_exposed, T0, u_tilde):
        """
        Eq. (14): ablative wall velocity for gold.

        Parameters are exposure time (s), surface temperature (HeV), and u_tilde.
        Returns km/s.
        """
        if t_exposed <= 0:
            return 0.0
        t_ns = t_exposed * 1e9
        return -510.1 * u_tilde * (T0 ** 0.716) * (t_ns ** 0.036)

    @staticmethod
    def ablation_velocity_cupper(t_exposed, T0, u_tilde):
        """
        Eq. (14): ablative wall velocity for copper.

        Parameters are exposure time (s), surface temperature (HeV), and u_tilde.
        Returns km/s.
        """
        if t_exposed <= 0:
            return 0.0
        t_ns = t_exposed * 1e9
        return -464 * u_tilde * (T0 ** 0.55) * (t_ns ** 0.0348)

    @staticmethod
    def compute_R_t(t, dt, t_heat, R0_cm, T0, R_array_prev, wall_material='Gold', u_tilde=0.05):
        """
        Compute radius profile R(t, x) per zone due to wall ablation.

        Returns
        -------
        np.ndarray
            Radius array R_i in cm.
        """
        R_array = []
        for i in range(len(z)):
            if t_heat[i] >= t:
                R_array.append(R0_cm)
            else:
                t_exp = t - t_heat[i]
                if wall_material == 'Gold':
                    u_kms_t = AblationModel.ablation_velocity_gold(t_exp, T0, u_tilde)
                    u_kms_t_minus_dt = AblationModel.ablation_velocity_gold(t_exp - dt, T0, u_tilde)
                elif wall_material == 'Cupper':
                    u_kms_t = AblationModel.ablation_velocity_cupper(t_exp, T0, u_tilde)
                    u_kms_t_minus_dt = AblationModel.ablation_velocity_cupper(t_exp - dt, T0, u_tilde)
                u_cms = u_kms_t * 1e5
                u_cms_minus_dt = u_kms_t_minus_dt * 1e5
                R_prev_i = R0_cm if (R_array_prev is None) else R_array_prev[i]
                R_i = R_prev_i - (np.abs(u_cms) + np.abs(u_cms_minus_dt)) / 2.0 * dt
                R_array.append(max(R_i, 0.0))
        if wall_material == 'Be':
            R_array = np.full_like(R_array, R0_cm)
        return np.array(R_array)

    @staticmethod
    def compute_rho_effective(R0_cm, R_array, xF):
        """
        Effective mean foam density up to the heat front, Eq. (17):
            rho_eff = rho0 * V0 / V(t)

        where:
            V0   = pi * R0^2 * xF
            V(t) = ∫_0^{xF} pi * R(t,x)^2 dx
        """
        volume = 0.0
        xF_index = np.searchsorted(z, xF)
        if xF_index == 0:
            return rho
        if xF_index >= len(z):
            xF_index = len(z) - 1
        original_volume = 0
        for i in range(xF_index + 1):
            volume += np.pi * (R_array[i] ** 2) * dz
            original_volume += np.pi * R0_cm**2 * dz
        if volume > original_volume:
            return rho
        return rho * original_volume / volume

    @staticmethod
    def mask_wall_cells_from_ablation(T_mesh, R_mesh, z_mesh, data_of_R, t_ref):
        """
        Return temperature mesh masked to foam-only cells and contour points for R(t,z).

        Cells with r > R_new(z,t) are gold and are masked out (set to NaN).

        Parameters
        ----------
        T_mesh : np.ndarray
            2D temperature field (Nz, Nr)
        R_mesh : np.ndarray
            2D radial coordinate grid (Nz, Nr)
        z_mesh : np.ndarray
            1D axial coordinate array (Nz,)
        data_of_R : dict
            Dictionary mapping time (float) to R_array (1D profile)
        t_ref : float
            Reference time for interpolation

        Returns
        -------
        tuple
            T_mesh_plot (NaN in gold cells), contour_r, contour_z
        """
        T_mesh_plot = T_mesh
        contour_r = None
        contour_z = None

        if data_of_R is None or len(data_of_R) == 0:
            return T_mesh_plot, contour_r, contour_z, None

        r_times = np.array(list(data_of_R.keys()), dtype=float)
        t_r = r_times[np.argmin(np.abs(r_times - t_ref))]
        r_profile = np.asarray(data_of_R[t_r], dtype=float)
        z_profile = np.asarray(z, dtype=float)

        if z_profile.size != r_profile.size or z_profile.size <= 1:
            return T_mesh_plot, contour_r, contour_z, T_mesh_plot_wall

        r_interp = np.interp(z_mesh, z_profile, r_profile, left=np.nan, right=np.nan)
        valid = np.isfinite(r_interp)
        if not np.any(valid):
            return T_mesh_plot, contour_r, contour_z

        r_interface = np.clip(r_interp, 0.0, R_cm)
        foam_mask = valid[:, None] & (R_mesh <= r_interface[:, None])
        wall_mask = valid[:, None] & (R_mesh > r_interface[:, None])
        T_mesh_plot_foam = np.where(foam_mask, T_mesh, np.nan)
        T_mesh_plot_wall = np.where(wall_mask, T_mesh, np.nan)

        contour_r = r_interface[valid]
        contour_z = z_mesh[valid]
        return T_mesh_plot_foam, contour_r, contour_z, T_mesh_plot_wall
