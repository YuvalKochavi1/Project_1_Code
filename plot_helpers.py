import numpy as np
from model_main import analytic_wave_front_dispatch
import matplotlib.pyplot as plt
from csv_helpers import *

def plot_analytic_if_available(x_vals, y_vals, *, label, linestyle="--", color=None):
    """Helper function to plot analytic curves if the data is available (i.e., not None). This is used to conditionally plot the analytic curves in the comparison plots without needing to check for None every time we call this function."""
    if y_vals is not None:
        plt.plot(x_vals, y_vals, linestyle=linestyle, label=label, color=color)

def plot_csv_series(path, *, y_scale=1.0, label=None, **plot_kwargs):
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
        analytic_positions_marshak,
        label="Analytic x_F(t) (Marshak BC)",
        linestyle="-",
        color='blue',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_2D,
        label=f"Analytic x_F(t) ({wall_material} Lost + ablation + varying rho)",
        linestyle="--",
        color='black',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_2D_lam_eff,
        label=f"Analytic x_F(t) ({wall_material} Lost + ablation + varying rho + lam_eff)",
        linestyle="--",
        color='purple',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_no_marshak,
        label="HR",
        linestyle="-",
        color='green',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_gold_loss,
        label=f"Analytic x_F(t) ({wall_material} Loss)",
        linestyle="--",
        color='orange',
    )
    plot_analytic_if_available(
        times_to_store,
        analytic_positions_ablation_const_rho,
        label=f"Analytic x_F(t) ({wall_material} Ablation Const rho)",
        linestyle="--",
        color='red',
    )


def plot_standard_surface_temperature_models(times_to_store, *, Ts_1D=None, Ts_2D=None):
    plot_analytic_if_available(
        times_to_store,
        Ts_1D,
        label="Analytic 1D Ts(t) (Marshak BC)",
        linestyle="--",
        color='blue',
    )
    plot_analytic_if_available(
        times_to_store,
        Ts_2D,
        label="Analytic 2D Ts(t) (Gold Lost BC)",
        linestyle="--",
        color='red',
    )


def compute_standard_analytic_front_series(times_to_store, *, wall_material = 'Gold', lam_eff_power=1):
    analytic_positions_no_marshak = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="no_marshak",
    )
    analytic_positions_marshak, Ts_1D, _, _, data_of_R_marshak, bessel_data = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak",
        vary_rho=False,
    )
    analytic_positions_2D, Ts_2D, E_out_2D, Ew_out_2D, data_of_R_2D, bessel_data_2D = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_ablation",
        vary_rho=True,
    )
    analytic_positions_2D_lam_eff, Ts_2D_lam_eff, E_out_2D_lam_eff, Ew_out_2D_lam_eff, data_of_R_2D_lam_eff, bessel_data_2D_lam_eff = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_ablation",
        vary_rho=True,
        lam_eff=True,
        power=lam_eff_power,
    )
    analytic_wave_front_marshak_gold_loss, Ts_marshak_gold_loss, E_out_gold_loss, Ew_out_gold_loss, data_of_R_gold_loss, bessel_data_gold_loss = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_wall_loss",
        vary_rho=False,
    )
    analytic_wave_front_ablation_const_rho, Ts_ablation_const_rho, E_out_ablation_const_rho, Ew_out_ablation_const_rho, data_of_R_ablation_const_rho, bessel_data_ablation_const_rho = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        wall_material=wall_material,
        mode="marshak_ablation",
        vary_rho=False,
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
