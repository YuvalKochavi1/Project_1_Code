from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
# BASE_DIR = C:\Users\TLP-001\Documents\GitHub\WeaponGroup\Project 1
FIGURES_DIR = BASE_DIR / "Figures_new"


def _get_parameters_context():
    import parameters
    return parameters.Experiment, parameters.Material


def _figures_dir():
    experiment, material = _get_parameters_context()
    return FIGURES_DIR / experiment / material

def ensure_figures_dir():
    os.makedirs(_figures_dir(), exist_ok=True)


def save_figure(filename, model1_5=False, model2_D=False, dpi=None, bbox_inches=None):
    ensure_figures_dir()
    relative_path = Path(filename)
    if model1_5:
        relative_path = Path("1.5 model") / relative_path
    elif model2_D:
        relative_path = Path("2D_shape") / relative_path

    save_path = _figures_dir() / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    plt.savefig(str(save_path), **save_kwargs)
    print(f"Saved figure: {save_path}")


def article_temperature_path(filename):
    experiment, material = _get_parameters_context()
    return BASE_DIR / "Data_new" / experiment / material / "article" / "Temperatures" / filename

def article_energy_path(filename):
    experiment, material = _get_parameters_context()
    return BASE_DIR / "Data_new" / experiment / material / "article" / "energies" / filename

def article_front_path(filename):
    experiment, material = _get_parameters_context()
    return BASE_DIR / "Data_new" / experiment / material / "article" / "fronts" / filename

def article_radius_path(filename):
    experiment, material = _get_parameters_context()
    return BASE_DIR / "Data_new" / experiment / material / "article" / "radius" / filename

def read_xy_csv(path):
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    return df["x"].to_numpy(), df["y"].to_numpy()


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


def plot_analytic_if_available(x_vals, y_vals, *, label, linestyle="--", color=None):
    if y_vals is not None:
        plt.plot(x_vals, y_vals, linestyle=linestyle, label=label, color=color)


def plot_csv_curves(curve_specs):
    for spec in curve_specs:
        plot_csv_series(
            spec["path"],
            y_scale=spec.get("y_scale", 1.0),
            label=spec.get("label"),
            linestyle=spec.get("linestyle", "-"),
            color=spec.get("color"),
        )


def plot_csv_errorbars(errorbar_specs):
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


def extract_front_positions(dispatch_output):
    if isinstance(dispatch_output, tuple):
        return dispatch_output[0]
    return dispatch_output