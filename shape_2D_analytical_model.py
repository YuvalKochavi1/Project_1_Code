import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Simple font configuration - just serif
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'axes.unicode_minus': False,
})
from scipy.interpolate import interp1d

from csv_helpers import save_figure
from parameters import L, R_cm, alpha, beta, z, r_grid, Experiment, Material, alpha_gold, beta_gold, alpha_copper, beta_copper, alpha_be, beta_be
from model_main import BASE_DIR
from scipy import special
from eigen_bessel_solver import kappa_roots
from colormaps import VISIT_LIKE_CMAP, PAPER_CMAP, PRR_BACK_CMAP, PRR_MOORE_CMAP

def _resolve_colormap_settings(
    color_style="default",
    *,
    cmap=None,
    vmin=None,
    vmax=None,
    cbar_ticks=None,
):
    """Resolve colormap and scale configuration for temperature heatmaps."""
    style = str(color_style).strip().lower()

    if Material == "C8H8":
        default_vmax = 1.50
        default_cbar_ticks = (0.00, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4)
    elif Material == "SiO2_copper":
        default_vmax = 1.75
        default_cbar_ticks = (0.00, 0.25, 0.5, 0.75, 1.00, 1.25, 1.5)
    else:
        default_vmax = 2.00
        default_cbar_ticks = (0.00, 0.25, 0.5, 0.75, 1.00, 1.25, 1.5, 1.75)

    style_presets = {
        "default": {
            "cmap": "Spectral_r",
            "vmin": 0.00,
            "vmax": default_vmax,
            "cbar_ticks": default_cbar_ticks,
        },
        # Visit-style appearance: blue -> cyan -> green -> yellow -> red.
        "visit": {
            "cmap": VISIT_LIKE_CMAP,
            "vmin": 0.00,
            "vmax": 2.5,
            # Matches 0, 63, 125, 187, 250 eV from the reference figure (converted to heV).
            "cbar_ticks": (0.00, 0.63, 1.25, 1.87, 2.50),
        },
        # Paper colorbar: blue/purple to dark red gradient extracted from paper figure.
        "paper": {
            "cmap": PAPER_CMAP,
            "vmin": 0.00,
            "vmax": 2.5,
            "cbar_ticks": (0.00, 0.63, 1.25, 1.87, 2.50),
        },
        # PRR back palette extracted from the provided RGB samples.
        "prr_back": {
            "cmap": PRR_BACK_CMAP,
            "vmin": 0.00,
            "vmax": 2.00,
            "cbar_ticks": (0.00, 0.63, 1.25, 1.87, 2.00),
        },
        # PRR Moore palette extracted from the provided RGB samples.
        "prr_moore": {
            "cmap": PRR_MOORE_CMAP,
            "vmin": 0.00,
            "vmax": 3.00,
            "cbar_ticks": (0.00, 0.74, 1.47, 2.21, 3.00),
        },
    }

    if style not in style_presets:
        print(f"Warning: unknown color_style '{color_style}', falling back to default. Available styles: 'default', 'visit', 'paper', 'prr_back', 'prr_moore'.")
        style = "default"

    resolved = style_presets[style].copy()

    if cmap is not None:
        resolved["cmap"] = cmap
    if vmin is not None:
        resolved["vmin"] = float(vmin)
    if vmax is not None:
        resolved["vmax"] = float(vmax)
    if cbar_ticks is not None:
        resolved["cbar_ticks"] = cbar_ticks

    return resolved


def _normalize_color_option(color_option=None, color_style="default"):
    """Return selected preset name while supporting legacy/new argument names."""
    if color_option is not None:
        return str(color_option).strip().lower()
    return str(color_style).strip().lower()


def _closest_time_data(bessel_data, target_time):
    numeric_times = []
    for k in bessel_data.keys():
        try:
            numeric_times.append(float(k))
        except (TypeError, ValueError):
            continue
    available_times = np.array(numeric_times, dtype=float)
    if available_times.size == 0:
        raise ValueError("bessel_data contains no numeric time keys.")
    t_closest = float(available_times[np.argmin(np.abs(available_times - float(target_time)))])
    return t_closest, bessel_data[t_closest]

def _load_simulated_front_csv(t_target):
    if abs(float(t_target) - 2.5) < 1e-9:
        filename = '2.5ns.csv'
    elif abs(float(t_target) - 3.72) < 1e-9:
        filename = '3.72ns.csv'
    else:
        filename = f'{int(round(float(t_target)))}ns.csv'

    candidate_paths = [
        BASE_DIR / '2D_simulation' / 'data' / filename,
        BASE_DIR / 'Data_new' / Experiment / Material / '2D_shape' / filename,
    ]

    for csv_path in candidate_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df['x'].to_numpy(), df['y'].to_numpy()

    searched = '\n'.join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        f"Could not find simulated front CSV '{filename}'. Searched:\n{searched}"
    )

def _compute_temperature_mesh(z_mesh, z_front_radial, T_surface, exponent):
    T_mesh = np.zeros((z_mesh.size, z_front_radial.size), dtype=float)
    for i_z, z_val in enumerate(z_mesh):
        active = (z_val < z_front_radial) & (z_front_radial > 0)
        if np.any(active):
            T_mesh[i_z, active] = T_surface * ((1.0 - z_val / z_front_radial[active]) ** exponent)
    return T_mesh


def _compute_wall_heyney_horizontal_profile(
    T_foam,
    foam_mask,
    wall_mask,
    r_mesh,
    exponent_wall,
    *,
    is_ablation=True,
    r_mesh_wall=None,
    penetration_radius_profile=None,
    shock_radius_profile=None,
    shock_plateau_fraction=0.07,
):
    """Build Henyey-like wall profile with explicit ablation/non-ablation behavior."""
    if r_mesh_wall is None:
        r_mesh_wall = r_mesh

    T_wall = np.full((T_foam.shape[0], r_mesh_wall.size), np.nan, dtype=float)
    eps = 1e-12
    j_rcm = int(np.searchsorted(r_mesh, R_cm, side='right') - 1)
    j_rcm = int(np.clip(j_rcm, 0, r_mesh.size - 1))

    for i_z in range(T_foam.shape[0]):
        if is_ablation:
            if foam_mask is None or foam_mask.shape != T_foam.shape:
                continue
            foam_idx = np.where(foam_mask[i_z])[0]
            if foam_idx.size == 0:
                continue
            j_interface = int(foam_idx[-1])
            if j_interface >= r_mesh.size:
                continue
            r_start = float(r_mesh[j_interface])
            if r_start >= R_cm - eps:
                continue
            Ts_local = float(T_foam[i_z, j_interface])
        else:
            # No ablation: start profile at fixed foam-gold boundary R_cm.
            r_start = float(R_cm)
            Ts_local = float(T_foam[i_z, j_rcm])

        if not np.isfinite(Ts_local) or Ts_local <= 0.0:
            continue

        r_stop = float(R_cm)
        if penetration_radius_profile is not None and i_z < penetration_radius_profile.size:
            r_stop_raw = float(penetration_radius_profile[i_z])
            # Accept either absolute radius or penetration depth profile.
            if r_stop_raw < R_cm - eps:
                r_stop_raw = R_cm + max(r_stop_raw, 0.0)
            r_stop = float(np.clip(r_stop_raw, R_cm, r_mesh_wall[-1]))

        if r_stop <= r_start + eps:
            continue

        shock_stop = None
        if shock_radius_profile is not None and i_z < shock_radius_profile.size:
            shock_stop_raw = float(shock_radius_profile[i_z]) # the shock wave stop which can (and will if not zero) be beyond the penetration stop
            if shock_stop_raw < R_cm - eps:
                shock_stop_raw = R_cm + max(shock_stop_raw, 0.0)
            shock_stop = float(np.clip(shock_stop_raw, r_stop, r_mesh_wall[-1]))

        if (
            is_ablation
            and
            wall_mask is not None
            and wall_mask.shape == T_foam.shape
            and penetration_radius_profile is None
            and r_mesh_wall.shape == r_mesh.shape
            and np.allclose(r_mesh_wall, r_mesh)
        ):
            wall_idx = np.where(wall_mask[i_z])[0]
        else:
            wall_idx = np.where((r_mesh_wall >= r_start) & (r_mesh_wall <= r_stop))[0]

        if wall_idx.size == 0:
            continue

        r_wall = r_mesh_wall[wall_idx]
        eta = (r_wall - r_start) / max(r_stop - r_start, eps)
        eta = np.clip(eta, 0.0, 1.0)
        T_wall_row = Ts_local * np.power(np.maximum(1.0 - eta, 0.0), exponent_wall)
        T_wall[i_z, wall_idx] = T_wall_row

        if shock_stop is not None and shock_stop > r_stop + eps:
            shock_idx = np.where((r_mesh_wall >= r_stop) & (r_mesh_wall <= shock_stop))[0]
            if shock_idx.size > 0:
                T_wall[i_z, shock_idx] = shock_plateau_fraction * Ts_local

    return T_wall


def _save_heatmap_2D_data(r_mesh, z_mesh, z_F_radial, T_mesh_plot, t_closest, r_mesh_foam=None, wall="Gold", flattop=False):
    """Save front shape and temperature mesh data to CSV files."""
    from pathlib import Path
    
    output_data_dir = Path(BASE_DIR) / "Data_new" / Experiment / Material / "2D_shape"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    # If z_F_radial is on a different grid (foam grid), interpolate to r_mesh
    if r_mesh_foam is not None and len(r_mesh_foam) == len(z_F_radial):
        z_F_radial_interp = np.interp(r_mesh, r_mesh_foam, z_F_radial, left=z_F_radial[0], right=z_F_radial[-1])
    else:
        z_F_radial_interp = z_F_radial
    
    # Save front shape (z_F as function of r)
    front_shape_df = pd.DataFrame({
        'r_cm': r_mesh,
        'z_F_radial_cm': z_F_radial_interp,
    })
    if flattop:
        front_shape_df.to_csv(output_data_dir / f"front_shape_{wall}_flattop_t{t_closest:.2f}ns.csv", index=False)
    else:   
        front_shape_df.to_csv(output_data_dir / f"front_shape_{wall}_t{t_closest:.2f}ns.csv", index=False)
    
    # Save temperature mesh
    T_mesh_df = pd.DataFrame(T_mesh_plot)
    T_mesh_df.to_csv(output_data_dir / f"T_mesh_{wall}_t{t_closest:.2f}ns.csv", index=False, header=False)
    # Save coordinate grids for reference
    coords_df = pd.DataFrame({
        'z_axis_cm': z_mesh,
    })
    coords_df.to_csv(output_data_dir / f"z_coordinates_{wall}.csv", index=False)
    
    r_coords_df = pd.DataFrame({'r_axis_cm': r_mesh})
    r_coords_df.to_csv(output_data_dir / f"r_coordinates_{wall}.csv", index=False)


def plot_2D_front_spatial(bessel_data, z_F_array, times_array, times_ns=[1.0, 2.0, 2.5]):
    """
    Plot z_F(r,t) in 2D spatial coordinates (r, z) showing the curved front
    from z=0 to z=L at different times.

    Parameters
    ----------
    bessel_data : dict
        Dictionary with time (ns) as keys, containing Bessel function data
    z_F_array : array
        Array of front positions z_F(t) in cm
    times_array : array
        Array of times (in ns) corresponding to z_F_array
    times_ns : list
        List of times (in ns) to plot
    """
    n_plots = len(times_ns)
    if n_plots == 1:
        fig_height = 6.5
        axes_height = fig_height - 1.5
        axes_width = axes_height * (R_cm / (L/2))
        fig_width = axes_width + 0.9 + 0.3
        wspace = 0.3
        fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, fig_height))
    else:
        fig_height = 6.0
        axes_height = fig_height - 1.5
        axes_width = axes_height * (R_cm / (L/2))
        wspace = 0.3
        fig_width = 0.9 + 0.3 + (n_plots + (n_plots - 1) * wspace) * axes_width
        fig, axes = plt.subplots(
            1, 
            n_plots, 
            figsize=(fig_width, fig_height), 
            gridspec_kw={"wspace": wspace}
        )
    if n_plots == 1:
        axes = [axes]

    for idx, t_target in enumerate(times_ns):
        if bessel_data and len(bessel_data) > 0:
            t_closest, data = _closest_time_data(bessel_data, t_target)
        else:
            # Fallback for modes that do not populate bessel_data (e.g. no wall-loss Marshak).
            t_closest = float(t_target)
            z_F_t = np.interp(t_closest, times_array, z_F_array)
            data = {
                'r_grid': np.asarray(r_grid, dtype=float),
                'r_gold_grid': np.asarray(r_grid, dtype=float),
                'z_grid': np.asarray(z, dtype=float),
                'z_F_radial': np.full_like(np.asarray(r_grid, dtype=float), z_F_t, dtype=float),
            }

        r_grid = data['r_grid']
        z_grid = data['z_grid']
        J0_profiles = data['J0_profiles']
        J0_profiles_approx = data['J0_profiles_approx']
        kappa_0 = data['kappa_0']
        kappa_0_approx = data['kappa_0_approx']
        albedo = data['albedo']

        # Get z_F(t) at this time
        z_F_t = np.interp(t_closest, times_array, z_F_array)

        z_F_radial = np.asarray(data['z_F_radial'], dtype=float)
        z_F_radial_approx = np.asarray(data['z_F_radial_approx'], dtype=float)

        ax = axes[idx]

        # Plot the domain boundaries
        # ax.axhline(y=0, color='black', linewidth=2, label='z = 0 (wall)')
        # ax.axhline(y=L, color='gray', linewidth=2, linestyle='--', label=f'z = L = {L:.2f} cm')
        # ax.axvline(x=0, color='black', linewidth=1, alpha=0.5, label='r = 0 (axis)')
        # ax.axvline(x=R_cm, color='gray', linewidth=2, linestyle='--', label=f'r = R = {R_cm:.2f} cm')

        # Plot the curved front position
        ax.plot(
            r_grid,
            z_F_radial,
            linewidth=3,
            color='red',
            label=r'$z_F(r,t)$',
            linestyle='-',
            markersize=4,
        )

        # r_csv, z_csv = _load_simulated_front_csv(t_target)
        # ax.plot(r_csv, z_csv, color='blue', label=f'Simulated front ({t_target:.2f} ns)', linestyle='--')

        # Shade the region behind the front (heated region)
        # ax.fill_between(r_grid, 0, z_F_radial, alpha=0.3, color='orange', label='Heated region')

        ax.set_xlabel(r'$r$ [cm]', fontsize=12, fontname='serif')
        if idx == 0:
            ax.set_ylabel(r'$z$ [cm]', fontsize=12, fontname='serif')
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        # ax.set_title(
        #     f't = {t_closest:.2f} ns\n$\kappa_0$ = {kappa_0:.3f}, $\kappa_0^{{approx}}$ = {kappa_0_approx:.3f}, albedo = {albedo:.3f}',
        #     fontsize=12,
        #     fontname='serif',
        # )
        ax.set_ylim([0, L/2])
        ax.set_xlim([0, R_cm])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best', prop={'family': 'serif'})
        ax.set_aspect('equal', adjustable='box')

    #plt.suptitle('2D Spatial View of Heat Front z_F(r,t) in Cylindrical Geometry', fontsize=14, y=1.00, fontname='serif')
    # plt.tight_layout()

    save_figure('2D_front_spatial.png', model2_D=True, dpi=150, bbox_inches='tight')
    plt.close()


def plot_temperature_heatmap_2D(
    bessel_data,
    z_F_array,
    T_s_array,
    times_array,
    times_ns=[1.0, 2.0, 2.5],
    ablation=False,
    wall="Gold",
    show_plot=True,
    title_suffix="",
    color_option=None,
    color_style="default",
    cmap=None,
    vmin=None,
    vmax=None,
    cbar_ticks=None,
    show_shock=True,
    flattop=False,
):
    """
    Plot 2D temperature heatmaps T(r,z,t) = T_s(t) * (1 - z/z_F(r,t))^(1/(4+alpha-beta))
    in cylindrical geometry for specified times.

    Parameters
    ----------
    bessel_data : dict
        Dictionary with time (ns) as keys, containing Bessel function data
    z_F_array : array
        Array of front positions z_F(t) in cm
    T_s_array : array
        Array of surface temperatures T_s(t) in heV
    times_array : array
        Array of times (in ns) corresponding to z_F_array and T_s_array
    times_ns : list
        List of times (in ns) to plot
    color_option : str, optional
        Preferred argument to choose colormap/scale preset.
        Supported: "default", "visit".
    color_style : str
        Backward-compatible alias for preset choice.
    cmap, vmin, vmax, cbar_ticks : optional
        Manual overrides for the preset values.
    show_shock : bool, optional
        If True (default), displays the shock boundary. If False, hides the shock.
    """
    # Get material parameters from global scope
    exponent = 1.0 / (4.0 + alpha - beta)  # Self-similar profile exponent
    if wall == "Gold":
        exponent_wall = 1.0 / (4.0 + alpha_gold - beta_gold)
    elif wall == "Copper":
        exponent_wall = 1.0 / (4.0 + alpha_copper - beta_copper)
    elif wall == "Be":
        exponent_wall = 1.0 / (4.0 + alpha_be - beta_be)
    elif wall == "Vacuum":
        exponent_wall = 0
    else:
        print(f"Warning: Unrecognized wall material '{wall}', using foam exponent for the wall heyney-like profile.")
        exponent_wall = exponent

    selected_color_style = _normalize_color_option(color_option=color_option, color_style=color_style)
    cmap_settings = _resolve_colormap_settings(
        selected_color_style,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_ticks=cbar_ticks,
    )

    plot_cmap = plt.get_cmap(cmap_settings['cmap']).copy()
    plot_cmap.set_bad('white')


    n_plots = len(times_ns)
    if bessel_data and len(bessel_data) > 0:
        first_time = next(iter(bessel_data))
        first_data = bessel_data[first_time]
        r_max = float(first_data.get('r_gold_grid', first_data.get('r_grid', [R_cm]))[-1])
    else:
        r_max = float(globals().get('r_gold', globals().get('r_grid', [R_cm]))[-1])
    
    if n_plots == 1:
        fig_height = 7.5
        axes_height = fig_height - 1.5
        axes_width = axes_height * (r_max / L)
        fig_width = axes_width + 0.9 + 1.2
        wspace = 0.35
        fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, fig_height))
    else:
        fig_height = 6.5
        axes_height = fig_height - 1.5
        axes_width = axes_height * (r_max / L)
        wspace = 0.35
        plot_cell_width = axes_width + 0.9
        fig_width = 0.9 + 0.3 + (n_plots * plot_cell_width) + (n_plots - 1) * wspace * plot_cell_width
        fig, axes = plt.subplots(
            1, 
            n_plots, 
            figsize=(fig_width, fig_height), 
            gridspec_kw={"wspace": wspace}
        )
    if n_plots == 1:
        axes = [axes]

    for idx, t_target in enumerate(times_ns):
        if bessel_data and len(bessel_data) > 0:
            t_closest, data = _closest_time_data(bessel_data, t_target)
        else:
            # Fallback for modes that do not populate bessel_data.
            t_closest = float(t_target)
            z_F_t = np.interp(t_closest, times_array, z_F_array)
            r_grid_base = np.asarray(globals()['r_grid'], dtype=float)
            data = {
                'r_grid': r_grid_base,
                'r_gold_grid': r_grid_base,
                'z_grid': np.asarray(z, dtype=float),
                'z_F_radial': np.full_like(r_grid_base, z_F_t, dtype=float),
            }

        r_grid = data['r_grid']
        z_grid = data['z_grid']

        # Get z_F(t) and T_s(t) at this time
        T_s_t = np.interp(t_closest, times_array, T_s_array)

        # Use foam grid for wave-front model and extended gold grid for plotting.
        r_mesh_foam = np.asarray(data.get('r_grid', r_grid), dtype=float)
        r_mesh = np.asarray(data.get('r_gold_grid', r_mesh_foam), dtype=float)
        z_mesh = z_grid
        R_mesh, Z_mesh = np.meshgrid(r_mesh, z_mesh) # 2D grid for plotting

        z_F_radial = np.asarray(data['z_F_radial'], dtype=float)
        T_mesh_foam = _compute_temperature_mesh(z_mesh, z_F_radial, T_s_t, exponent)

        # Map foam solution onto full radial grid; cold/unreached gold is set to 0.
        T_mesh_plot = np.zeros((z_mesh.size, r_mesh.size), dtype=float)
        foam_domain = r_mesh <= R_cm
        for i_z in range(z_mesh.size):
                T_mesh_plot[i_z, foam_domain] = np.interp(
                r_mesh[foam_domain],
                r_mesh_foam,
                T_mesh_foam[i_z],
            left=0.0,
            right=0.0,
            )

        ablation_contour_r = None
        ablation_contour_z = None

        penetration_profile = data.get('wall_penetration_radius_profile')
        if penetration_profile is None:
            penetration_profile = np.full_like(z_mesh, R_cm, dtype=float)
        else:
            penetration_profile = np.asarray(penetration_profile, dtype=float)
            if penetration_profile.shape != z_mesh.shape:
                penetration_profile = np.full_like(z_mesh, R_cm, dtype=float)

        shock_profile = data.get('shock_penetration_radius_profile')
        if shock_profile is not None:
            shock_profile = np.asarray(shock_profile, dtype=float)
            if shock_profile.shape != z_mesh.shape:
                shock_profile = None

        if ablation:
            ablation_contour_r = data.get('ablation_contour_r')
            ablation_contour_z = data.get('ablation_contour_z')

        foam_mask = data.get('ablation_foam_mask')
        wall_mask = data.get('ablation_wall_mask')
        if foam_mask is not None:
            foam_mask = np.asarray(foam_mask, dtype=bool)
        if wall_mask is not None:
            wall_mask = np.asarray(wall_mask, dtype=bool)

        if ablation:
            if (
                foam_mask is not None
                and wall_mask is not None
                and foam_mask.shape == T_mesh_foam.shape
                and wall_mask.shape == T_mesh_foam.shape
            ):
                T_wall_profile = _compute_wall_heyney_horizontal_profile(
                    T_mesh_foam,
                    foam_mask,
                    wall_mask,
                    r_mesh_foam,
                    exponent_wall,
                    is_ablation=True,
                    r_mesh_wall=r_mesh,
                    penetration_radius_profile=penetration_profile,
                    shock_radius_profile=shock_profile,
                )
                wall_valid = np.isfinite(T_wall_profile)
                if np.any(wall_valid):
                    T_mesh_plot[wall_valid] = T_wall_profile[wall_valid]
        else:
            if wall != "Vacuum":
                # Non-ablation: wall heat wave starts at fixed R_cm and penetrates into gold.
                T_wall_profile = _compute_wall_heyney_horizontal_profile(
                    T_mesh_foam,
                    foam_mask=None,
                    wall_mask=None,
                    r_mesh=r_mesh_foam,
                    exponent_wall=exponent_wall,
                    is_ablation=False,
                    r_mesh_wall=r_mesh,
                    penetration_radius_profile=penetration_profile,
                    shock_radius_profile=shock_profile,
                )
                wall_valid = np.isfinite(T_wall_profile)
                if np.any(wall_valid):
                    T_mesh_plot[wall_valid] = T_wall_profile[wall_valid]

        if show_shock and shock_profile is not None:
            shock_mask = np.isfinite(shock_profile)
            if np.any(shock_mask):
                T_mesh_plot = np.array(T_mesh_plot, copy=True)
                for i_z, shock_r in enumerate(shock_profile):
                    if np.isfinite(shock_r):
                        T_mesh_plot[i_z, r_mesh > shock_r] = np.nan

        # Save front shape and temperature mesh data before plotting
        _save_heatmap_2D_data(r_mesh, z_mesh, z_F_radial, T_mesh_plot, t_target, r_mesh_foam=r_mesh_foam, wall=wall, flattop=flattop)

        ax = axes[idx]

        # Create heatmap with gouraud shading
        pcm = ax.pcolormesh(
            R_mesh,
            Z_mesh,
            T_mesh_plot,
            shading='gouraud',
            cmap=plot_cmap,
            vmin=cmap_settings['vmin'],
            vmax=cmap_settings['vmax'],
        )

        # Add contour lines with temperature labels in eV
        # contour_levels = np.array([1, 1.2, 1.3, 1.4, 1.5])
        # contour = ax.contour(
        #     R_mesh,
        #     Z_mesh,
        #     T_mesh_plot,
        #     levels=contour_levels,
        #     colors='black',
        #     linewidths=0.5,
        #     alpha=0.3,
        # )
        # ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

        if Material == "SiO2" and ablation:
            if (
                ablation_contour_r is not None
                and ablation_contour_z is not None
                and len(ablation_contour_r) > 1
            ):
                ax.plot(
                    ablation_contour_r,
                    ablation_contour_z,
                    color='black',
                    linewidth=0.5,
                    linestyle='-',
                )

        # Plot the front position
        #ax.plot(r_mesh_foam, z_F_radial, linewidth=3, color='cyan', label='Front z_F(r,t)', linestyle='--')

        # Add colorbar
        use_right_layout = (Material == "C8H7Cl") or (Material == "SiO2" and ablation) or (Material == "C8H8")
        if not use_right_layout:
            cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
            if cmap_settings['cbar_ticks'] is not None:
                cbar.set_ticks(cmap_settings['cbar_ticks'])
            cbar.set_label(r'$T$ [heV]')
        else:
            cbar = plt.colorbar(pcm, ax=ax, pad=0.3, fraction=0.046)
            if cmap_settings['cbar_ticks'] is not None:
                cbar.set_ticks(cmap_settings['cbar_ticks'])
            cbar.set_label(r'$T$ [heV]')

        # Domain boundaries
        # ax.axhline(y=0, color='white', linewidth=2, linestyle='-', alpha=0.7)
        # ax.axhline(y=L/2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        # ax.axvline(x=R_cm, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$r$ [cm]')
        if not use_right_layout:
            if idx == 0:
                ax.set_ylabel(r'$z$ [cm]')
                ax.tick_params(labelleft=True)
            else:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False, labelright=False)
        else:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(r'$z$ [cm]')
            ax.tick_params(labelleft=False, labelright=True)
        # ax.set_title(r'$t = ' + f'{t_closest:.2f}' + r'$ ns')
        print(f"Plotting time {t_closest:.2f} ns, t_target was {t_target:.2f} ns")
        ax.set_ylim([0, L])
        if r_mesh[-1] > R_cm and Material != "C8H8":
            ax.set_xlim([0.0, float(r_mesh[-1])])
        else:
            ax.set_xlim([0.0, R_cm])
        ax.set_aspect('equal', adjustable='box')

    title_fontsize = 12 if n_plots == 1 else 16
    # plt.suptitle(
    #     f'Temperature Distribution T(r,z,t) with Self-Similar Profile (exponent = {exponent:.3f})',
    #     y=0.98 if n_plots == 1 else 1.00,
    #     fontsize=title_fontsize,
    # )
    fig.subplots_adjust(wspace=wspace if n_plots > 1 else 0.2)
    # plt.tight_layout(pad=0.6, w_pad=0.2)
    plt.show()
    save_figure(f'temperature_heatmap_2D{title_suffix}.png', model2_D=True, dpi=150, bbox_inches='tight')
    if show_plot:
        backend = plt.get_backend().lower()
        if 'agg' not in backend:
            plt.show()
    #plt.close()


#########################################
# French analytical model
#########################################

def compute_T4_2D_series_from_front(
    r_mesh,
    z_mesh,
    z_F_t,
    T_s_t,
    *,
    epsilon=0.4,
    R_value=R_cm,
    n_terms=5,
):
    """
    Compute T^4(r,z,t) from the analytical cylindrical series model using z_F(t).

    Implements
        T^4 / T_s0^4 = -(1 + epsilon/2) * J0(sqrt(2*epsilon) * r / R)
                       * sinh(k0 * (z - z_F)) / sinh(k0 * z_F)
                       - 4*epsilon * sum_n [ J0(kn*r)/(alpha_{1,n}^2 * J0(alpha_{1,n}))
                                             * sinh(kn*(z-z_F))/sinh(kn*z_F) ]

    where k0 = sqrt(2*epsilon)/R and kn = alpha_{1,n}/R.
    """
    r_mesh = np.asarray(r_mesh, dtype=float)
    z_mesh = np.asarray(z_mesh, dtype=float)

    Rm, Zm = np.meshgrid(r_mesh, z_mesh)
    zf_safe = max(float(z_F_t), 1e-12)

    k0 = np.sqrt(2.0 * epsilon) / R_value
    denom0 = np.sinh(k0 * zf_safe)
    if abs(denom0) < 1e-14:
        denom0 = np.sign(denom0) * 1e-14 if denom0 != 0 else 1e-14

    term0 = -(
        (1.0 + 0.5 * epsilon)
        * special.j0(np.sqrt(2.0 * epsilon) * Rm / R_value)
        * np.sinh(k0 * (Zm - zf_safe))
        / denom0
    )

    alpha_1n = special.jn_zeros(1, int(n_terms))
    sum_term = np.zeros_like(term0)
    for i in range(1, len(alpha_1n)):
        kn = kappa_roots(epsilon, R_cm, n_terms)[i]
        denom_n = np.sinh(kn * zf_safe)
        if abs(denom_n) < 1e-14:
            denom_n = np.sign(denom_n) * 1e-14 if denom_n != 0 else 1e-14

        bessel_den = (alpha_1n[i] ** 2) * special.j0(alpha_1n[i])
        if abs(bessel_den) < 1e-14:
            continue

        sum_term += (
            special.j0(kn * Rm)
            * np.sinh(kn * (Zm - zf_safe))
            / (denom_n * bessel_den)
        )

    ratio = term0 - 4.0 * epsilon * sum_term

    # Physical domain used here: ahead of front is set to zero.
    ratio = np.where(Zm <= zf_safe, ratio, 0.0)
    ratio = np.maximum(ratio, 0.0)
    T4 = (float(T_s_t) ** 4) * ratio
    return T4


def plot_temperature_heatmap_2D_series_model(
    z_F_array,
    T_s_array,
    times_array,
    *,
    bessel_data=None,
    data_of_R=None,
    times_ns=(1.0, 2.0, 2.5),
    epsilon=0.4,
    n_terms=40,
    n_r=100,
    n_z=200,
    color_option=None,
    color_style="default",
    cmap=None,
    vmin=None,
    vmax=None,
    cbar_ticks=None,
):
    """
    Plot field from compute_T4_2D_series_from_front for selected times.

    If plot_T4 is True, plots T^4 directly; otherwise plots T = (T^4)^(1/4).
    """
    selected_color_style = _normalize_color_option(color_option=color_option, color_style=color_style)
    cmap_settings = _resolve_colormap_settings(
        selected_color_style,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_ticks=cbar_ticks,
    )

    fig, axes = plt.subplots(1, len(times_ns), figsize=(18, 6))
    if len(times_ns) == 1:
        axes = [axes]

    for idx, t_target in enumerate(times_ns):
        t_eval = float(t_target)
        z_F_t = np.interp(t_eval, times_array, z_F_array)
        T_s_t = np.interp(t_eval, times_array, T_s_array)

        # Use global grids from parameters
        r_mesh = r_grid
        z_mesh = z
        R_mesh, Z_mesh = np.meshgrid(r_mesh, z_mesh)

        T4_mesh = compute_T4_2D_series_from_front(
            r_mesh,
            z_mesh,
            z_F_t,
            T_s_t,
            epsilon=epsilon,
            R_value=R_cm,
            n_terms=n_terms,
        )

        field_mesh = np.power(np.maximum(T4_mesh, 0.0), 0.25)
        field_label = r'$T$ [heV]'

        ax = axes[idx]
        pcm = ax.pcolormesh(
            R_mesh,
            Z_mesh,
            field_mesh,
            shading='gouraud',
            cmap=cmap_settings['cmap'],
            vmin=cmap_settings['vmin'],
            vmax=cmap_settings['vmax'],
        )

        if data_of_R is not None and len(data_of_R) > 0:
            r_times = np.array(list(data_of_R.keys()), dtype=float)
            t_r = r_times[np.argmin(np.abs(r_times - t_eval))]
            r_profile = np.asarray(data_of_R[t_r], dtype=float)
            z_profile = np.asarray(z, dtype=float)

            if z_profile.size == r_profile.size and z_profile.size > 1:
                r_interp = np.interp(z_mesh, z_profile, r_profile, left=np.nan, right=np.nan)
                valid = np.isfinite(r_interp)
                if np.any(valid):
                    r_interp_plot = np.clip(r_interp[valid], 0.0, R_cm)
                    # ax.plot(r_interp_plot, z_mesh[valid], color='white', linewidth=2.5,
                    #         linestyle='-', label='Ablation contour R(t,z)')

        cbar = plt.colorbar(pcm, ax=ax)
        if cmap_settings['cbar_ticks'] is not None:
            cbar.set_ticks(cmap_settings['cbar_ticks'])
        cbar.set_label(field_label)

        ax.axhline(y=0, color='white', linewidth=2, linestyle='-', alpha=0.7)
        ax.axhline(y=L, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(x=R_cm, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$r$ [cm]')
        if idx == 0:
            ax.set_ylabel(r'$z$ [cm]')
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        ax.set_title(f't = {t_eval:.2f} ns')
        ax.set_ylim([0, L/2])
        ax.set_xlim([0, R_cm])
        ax.legend(fontsize=9, loc='upper right')
        ax.set_aspect('auto')

    title_prefix = 'Temperature T'
    plt.suptitle(
        f'{title_prefix}(r,z,t) from cylindrical series model (epsilon={epsilon:.3f}, terms={int(n_terms)})',
        y=1.00,
        fontsize=18,
    )
    plt.tight_layout()

    output_name = 'temperature_heatmap_2D_series_model.png'
    save_figure(output_name, model2_D=True, dpi=150, bbox_inches='tight')
    plt.close()