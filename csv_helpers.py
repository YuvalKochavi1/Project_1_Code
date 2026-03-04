from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def extract_front_positions(dispatch_output):
    if isinstance(dispatch_output, tuple):
        return dispatch_output[0]
    return dispatch_output

def export_analytic_positions_csv(times_to_store, grouped_series, output_csv_path):
    """
    Exports analytic positions to a CSV file. The `grouped_series` should be a dictionary where keys are group names (e.g., "Marshak", "2D Shape") and values are dictionaries mapping series names (e.g., "Analytic Front", "Ts") to their corresponding lists of values.
    """
    times_ns = np.asarray(times_to_store)
    combined = {"time_ns": times_ns}

    for group_name, series_dict in grouped_series.items():
        for series_name, series_values in series_dict.items():
            if series_values is None:
                continue

            series_array = np.asarray(series_values)
            if series_array.shape[0] != times_ns.shape[0]:
                raise ValueError(
                    f"Series '{series_name}' length ({series_array.shape[0]}) does not match time array length ({times_ns.shape[0]})."
                )

            column_name = f"{group_name} ({series_name})" if group_name else series_name
            combined[column_name] = series_array

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(combined).to_csv(output_csv_path, index=False)
    print(f"Saved analytic positions CSV: {output_csv_path}")