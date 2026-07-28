import numpy as np
from model_main import analytic_wave_front_dispatch
import matplotlib.pyplot as plt
from csv_helpers import *
from shape_2D_analytical_model import _closest_time_data

def plot_analytic_if_available(x_vals, y_vals, *, label, linestyle="--", color=None):
    """Helper function to plot analytic curves if the data is available (i.e., not None). This is used to conditionally plot the analytic curves in the comparison plots without needing to check for None every time we call this function."""
    if y_vals is not None:
        plt.plot(x_vals, y_vals, linestyle=linestyle, label=label, color=color)

def plot_csv_series(path, *, y_scale=1, label=None, **plot_kwargs):
    x_vals, y_vals = read_xy_csv(path)
    plt.plot(x_vals, y_vals / y_scale, label=label, **plot_kwargs)


def plot_csv_errorbar(path, *, y_scale=1.0, label=None, xerr=None, yerr=None, **errorbar_kwargs):
    x_vals, y_vals = read_xy_csv(path)
    plt.errorbar(
        x_vals,
        y_vals / y_scale,
        xerr=xerr,
        yerr=yerr,
        label=label,
        **errorbar_kwargs,
    )


def plot_csv_curves(curve_specs):
    """Plots multiple curves from CSV files. Each spec in `curve_specs` should be a dictionary containing at least the 'path' key, and optionally 'y_scale', 'label', 'linestyle', and 'color' keys for customizing the plot. used mostly for plotting the articles' curves."""
    for spec in curve_specs:
        plot_csv_series(
            spec["path"],
            y_scale=spec.get("y_scale", 1.0),
            label=spec.get("label"),
            linestyle=spec.get("linestyle", "-"),
            color=spec.get("color"),
        )


def plot_csv_errorbars(errorbar_specs):
    """Plots error bars from CSV files. Each spec in `errorbar_specs` should be a dictionary containing at least the 'path' key, and optionally 'y_scale', 'label', 'xerr', 'yerr', 'fmt', 'capsize', 'elinewidth', 'markersize', and 'color' keys for customizing the error bars. used mostly for plotting the experimental data with error bars."""
    for spec in errorbar_specs:
        plot_csv_errorbar(
            spec["path"],
            y_scale=spec.get("y_scale", 1.0),
            label=spec.get("label"),
            xerr=spec.get("xerr"),
            yerr=spec.get("yerr"),
            fmt=spec.get("fmt", "o"),
            capsize=spec.get("capsize", 4),
            elinewidth=spec.get("elinewidth", 1.5),
            markersize=spec.get("markersize", 8),
            color=spec.get("color"),
        )


def plot_standard_front_analytic_models(
    times_to_store,
    *,
    analytic_positions_marshak=None,
    analytic_positions_2D=None,
    analytic_positions_2D_lam_eff=None,
    analytic_positions_no_marshak=None,
    analytic_positions_gold_loss=None,
    analytic_positions_ablation_const_rho=None,
    wall_material = 'Gold',
):
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_no_marshak,
        label=r"$\mathrm{HR}$",
        linestyle="-",
        color='green',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_marshak,
        label=r"Model - Marshak BC",
        linestyle="-",
        color='blue',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_gold_loss,
        label=fr"Model - {wall_material} Loss",
        linestyle="-",
        color='orange',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_2D,
        label=fr"Model - 2D effects",
        linestyle="-",
        color='black',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_2D_lam_eff,
        label=fr"Model - 2D effects + $\lambda_{{\mathrm{{eff}}}}$",
        linestyle="-",
        color='red',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_ablation_const_rho,
        label=fr"Model - ({wall_material} Ablation Const $\rho$)",
        linestyle="-",
        color='pink',
    )


def plot_standard_surface_temperature_models(times_to_store, *, Ts_1D=None, Ts_2D=None, Ts_2D_lam_eff=None):
    plot_analytic_if_available(
        times_to_store,
        Ts_1D,
        label=r"Model $1\mathrm{D}$ $T_s(t)$ (Marshak BC)",
        linestyle="--",
        color='blue',
    )
    plot_analytic_if_available(
        times_to_store,
        Ts_2D,
        label=r"Model $2\mathrm{D}$ $T_s(t)$ (Gold Lost BC)",
        linestyle="--",
        color='red',
    )
    plot_analytic_if_available(
        times_to_store,
        Ts_2D_lam_eff,
        label=r"Model $2\mathrm{D}$ $T_s(t)$ (Gold Lost BC + $\lambda_{{\mathrm{{eff}}}}$)",
        linestyle="--",
        color='green',
    )


def compute_standard_analytic_front_series(times_to_store, *, wall_material = 'Gold', lam_eff_power=1, g_gold_scale=1.0):
    analytic_positions_no_marshak = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="no_marshak",
        g_gold_scale=g_gold_scale,
    )
    analytic_positions_marshak, Ts_1D, E_marshak, _, data_of_R_marshak, bessel_data = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak",
        vary_rho=False,
        g_gold_scale=g_gold_scale,
    )
    analytic_positions_2D, Ts_2D, E_out_2D, Ew_out_2D, data_of_R_2D, bessel_data_2D = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_ablation",
        vary_rho=True,
        g_gold_scale=g_gold_scale,
    )
    analytic_positions_2D_lam_eff, Ts_2D_lam_eff, E_out_2D_lam_eff, Ew_out_2D_lam_eff, data_of_R_2D_lam_eff, bessel_data_2D_lam_eff = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_ablation",
        vary_rho=True,
        lam_eff=True,
        power=lam_eff_power,
        g_gold_scale=g_gold_scale,
    )
    analytic_wave_front_marshak_gold_loss, Ts_marshak_gold_loss, E_out_gold_loss, Ew_out_gold_loss, data_of_R_gold_loss, bessel_data_gold_loss = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_wall_loss",
        vary_rho=False,
        g_gold_scale=g_gold_scale,
    )
    analytic_wave_front_ablation_const_rho, Ts_ablation_const_rho, E_out_ablation_const_rho, Ew_out_ablation_const_rho, data_of_R_ablation_const_rho, bessel_data_ablation_const_rho = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_ablation",
        vary_rho=False,
        g_gold_scale=g_gold_scale,
    )
    return {
        "analytic_positions_no_marshak": analytic_positions_no_marshak,
        "analytic_positions_marshak": analytic_positions_marshak,
        "analytic_positions_2D": analytic_positions_2D,
        "analytic_positions_2D_lam_eff": analytic_positions_2D_lam_eff,
        "analytic_positions_gold_loss": analytic_wave_front_marshak_gold_loss,
        "analytic_positions_ablation_const_rho": analytic_wave_front_ablation_const_rho,
        "Ts_1D": Ts_1D,
        "Ts_2D": Ts_2D,
        "Ts_2D_lam_eff": Ts_2D_lam_eff,
        "Ts_marshak_gold_loss": Ts_marshak_gold_loss,
        "Ts_ablation_const_rho": Ts_ablation_const_rho,
        "E_2D": E_out_2D,
        "E_wall_out_2D": Ew_out_2D,
        "E_2D_lam_eff": E_out_2D_lam_eff,
        "E_gold_loss": E_out_gold_loss,
        "E_marshak": E_marshak,
        "E_W_gold_loss": Ew_out_gold_loss,
        "E_ablation_const_rho": E_out_ablation_const_rho,
        "data_of_R_marshak": data_of_R_marshak,
        "bessel_data_marshak": bessel_data,
        "data_of_R_2D": data_of_R_2D,
        "bessel_data_2D": bessel_data_2D,
        "data_of_R_2D_lam_eff": data_of_R_2D_lam_eff,
        "bessel_data_2D_lam_eff": bessel_data_2D_lam_eff,
        "data_of_R_gold_loss": data_of_R_gold_loss,
        "bessel_data_gold_loss": bessel_data_gold_loss,
        "data_of_R_ablation_const_rho": data_of_R_ablation_const_rho,
        "bessel_data_ablation_const_rho": bessel_data_ablation_const_rho,
    }


def plot_albedo_arrays(bessel_data, z_grid=None, title="Albedo Profiles vs Depth", figsize=(18, 6), times_ns=None):
    """
    Plot albedo arrays for selected times in bessel_data in separate subplots.
    
    Parameters
    ----------
    bessel_data : dict
        Dictionary with keys as time (ns) and values as snapshots containing 'albedo_array' and 'avg_albedo'
    z_grid : array-like, optional
        Spatial grid (z values). If None, indices are used on x-axis.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    times_ns : list, optional
        List of specific times (in ns) to plot. If None, all times are plotted.
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects
    """
    if not bessel_data or len(bessel_data) == 0:
        print("Warning: bessel_data is empty, cannot plot albedo arrays")
        return None, None
    
    # Default to plotting first 3 times if not specified
    if times_ns is None:
        all_times_ns = sorted(bessel_data.keys())
        times_ns = all_times_ns[:min(3, len(all_times_ns))]
    
    fig, axes = plt.subplots(1, len(times_ns), figsize=figsize)
    if len(times_ns) == 1:
        axes = [axes]
    
    for idx, t_target in enumerate(times_ns):
        # Find closest time in bessel_data
        t_closest, data = _closest_time_data(bessel_data, t_target)
        
        if 'albedo_array' in data and 'avg_albedo' in data:
            albedo_array = data['albedo_array']
            avg_albedo = data['avg_albedo']
            
            # Use provided z_grid or indices
            if z_grid is not None:
                x_vals = z_grid
            else:
                x_vals = np.arange(len(albedo_array))
            
            ax = axes[idx]
            ax.plot(x_vals, albedo_array, marker='o', markersize=6, 
                   color='steelblue', linewidth=2.0)
            
            ax.set_xlabel('Depth (z)', fontsize=11)
            ax.set_ylabel('Albedo (a)', fontsize=11)
            ax.set_title(f't = {t_closest:.2f} ns\n⟨a⟩ = {avg_albedo:.3f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.0])
            
            print(f"Plotting albedo at time {t_closest:.2f} ns, t_target was {t_target:.2f} ns")
    
    fig.suptitle(title, fontsize=14, y=1.00)
    fig.tight_layout()
    
    return fig, axes
