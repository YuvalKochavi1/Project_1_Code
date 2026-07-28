import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import erf


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILES = [
    "z_110.csv",
    "z_180.csv",
    "z_330.csv",
    "z_590.csv",
    "z_840.csv",
]

RESULTS_FILE = "front_inflection_results.csv"
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# FIT MODEL
# ============================================================

def erf_rising_edge(
    time_ps,
    baseline,
    amplitude,
    t_front_ps,
    sigma_ps,
):
    """
    Error-function fit to the rising edge.

    t_front_ps is exactly the fitted inflection point.
    """
    argument = (
        time_ps - t_front_ps
    ) / (
        np.sqrt(2.0) * sigma_ps
    )

    return baseline + 0.5 * amplitude * (
        1.0 + erf(argument)
    )


# ============================================================
# FILE HANDLING
# ============================================================

def extract_distance_from_filename(filename):
    """
    Extract distance from names such as:
        z_110.csv
        profile_z_330.csv
    """
    stem = Path(filename).stem

    match = re.search(
        r"z[_=\- ]*(\d+(?:\.\d+)?)",
        stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not extract z distance from '{filename}'. "
            "Use a filename such as z_110.csv."
        )

    return float(match.group(1))


def load_profile(filename):
    """
    Load a CSV of the form:

        index,x,y

    x = time in ps
    y = measured signal
    """
    csv_path = Path(filename)
    if not csv_path.is_absolute():
        csv_path = SCRIPT_DIR / csv_path

    data = pd.read_csv(csv_path)

    required_columns = {"x", "y"}

    if not required_columns.issubset(data.columns):
        raise KeyError(
            f"{csv_path} must contain columns 'x' and 'y'. "
            f"Found: {list(data.columns)}"
        )

    time_ps = pd.to_numeric(
        data["x"],
        errors="coerce",
    ).to_numpy()

    signal = pd.to_numeric(
        data["y"],
        errors="coerce",
    ).to_numpy()

    valid = (
        np.isfinite(time_ps)
        & np.isfinite(signal)
    )

    time_ps = time_ps[valid]
    signal = signal[valid]

    if len(time_ps) < 10:
        raise ValueError(
            f"{csv_path} contains too few valid points."
        )

    # Sort by time
    order = np.argsort(time_ps)

    time_ps = time_ps[order]
    signal = signal[order]

    # Average values if duplicate times exist
    cleaned = pd.DataFrame({
        "time_ps": time_ps,
        "signal": signal,
    })

    cleaned = (
        cleaned
        .groupby("time_ps", as_index=False)
        .mean()
    )

    return (
        cleaned["time_ps"].to_numpy(),
        cleaned["signal"].to_numpy(),
    )


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def estimate_baseline(signal):
    """
    Estimate baseline from the first 8% of the signal.
    """
    number_of_points = max(
        5,
        int(0.08 * len(signal)),
    )

    return float(
        np.median(signal[:number_of_points])
    )


def get_savgol_window(
    number_of_points,
    requested_window=31,
    polynomial_order=3,
):
    """
    Find a valid odd Savitzky-Golay window.
    """
    window = min(
        requested_window,
        number_of_points,
    )

    if window % 2 == 0:
        window -= 1

    if window <= polynomial_order:
        return None

    return window


def smooth_signal(
    signal,
    requested_window=31,
    polynomial_order=3,
):
    window = get_savgol_window(
        number_of_points=len(signal),
        requested_window=requested_window,
        polynomial_order=polynomial_order,
    )

    if window is None:
        return signal.copy()

    return savgol_filter(
        signal,
        window_length=window,
        polyorder=polynomial_order,
    )


def select_rising_edge(
    time_ps,
    signal,
    lower_fraction=0.05,
    upper_fraction=0.90,
):
    """
    Select the rising edge between approximately 5% and 90%
    of the signal maximum, before the pulse peak.
    """
    smoothed = smooth_signal(signal)
    baseline = estimate_baseline(signal)

    peak_index = int(np.argmax(smoothed))
    peak_signal = smoothed[peak_index]

    amplitude = peak_signal - baseline

    if amplitude <= 0:
        raise ValueError(
            "No positive rising pulse was found."
        )

    lower_level = (
        baseline
        + lower_fraction * amplitude
    )

    upper_level = (
        baseline
        + upper_fraction * amplitude
    )

    all_indices = np.arange(len(signal))

    mask = (
        (all_indices <= peak_index)
        & (smoothed >= lower_level)
        & (smoothed <= upper_level)
    )

    selected = np.where(mask)[0]

    if len(selected) < 5:
        raise ValueError(
            "Too few points were found on the rising edge."
        )

    padding = max(
        3,
        int(0.02 * len(signal)),
    )

    start = max(
        0,
        selected[0] - padding,
    )

    stop = min(
        len(signal),
        selected[-1] + padding + 1,
    )

    return np.arange(start, stop)


# ============================================================
# ERROR-FUNCTION FIT
# ============================================================

def fit_inflection_point(time_ps, signal):
    """
    Fit the rising edge and return its inflection point.
    """
    fit_indices = select_rising_edge(
        time_ps,
        signal,
    )

    fit_time = time_ps[fit_indices]
    fit_signal = signal[fit_indices]

    smoothed = smooth_signal(fit_signal)
    baseline_guess = estimate_baseline(signal)

    amplitude_guess = (
        np.max(smoothed)
        - baseline_guess
    )

    derivative = np.gradient(
        smoothed,
        fit_time,
    )

    t_front_guess = fit_time[
        np.argmax(derivative)
    ]

    dt = np.median(
        np.diff(time_ps)
    )

    sigma_guess = max(
        3.0 * dt,
        (
            fit_time[-1]
            - fit_time[0]
        ) / 8.0,
    )

    initial_guess = [
        baseline_guess,
        amplitude_guess,
        t_front_guess,
        sigma_guess,
    ]

    signal_min = np.min(signal)
    signal_max = np.max(signal)
    signal_range = signal_max - signal_min

    lower_bounds = [
        signal_min - signal_range,
        0.0,
        fit_time[0],
        max(dt / 10.0, 1e-12),
    ]

    upper_bounds = [
        signal_max,
        3.0 * max(signal_range, 1e-12),
        fit_time[-1],
        time_ps[-1] - time_ps[0],
    ]

    parameters, covariance = curve_fit(
        erf_rising_edge,
        fit_time,
        fit_signal,
        p0=initial_guess,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        maxfev=100_000,
    )

    errors = np.sqrt(
        np.diag(covariance)
    )

    (
        baseline,
        amplitude,
        t_front_ps,
        sigma_ps,
    ) = parameters

    fitted_signal = erf_rising_edge(
        fit_time,
        *parameters,
    )

    residuals = (
        fit_signal
        - fitted_signal
    )

    ss_residual = np.sum(
        residuals**2
    )

    ss_total = np.sum(
        (
            fit_signal
            - np.mean(fit_signal)
        )**2
    )

    if ss_total > 0:
        r_squared = (
            1.0
            - ss_residual / ss_total
        )
    else:
        r_squared = np.nan

    signal_at_inflection = (
        baseline
        + 0.5 * amplitude
    )

    observed_maximum = np.max(signal)

    fraction_of_maximum = (
        signal_at_inflection - baseline
    ) / (
        observed_maximum - baseline
    )

    return {
        "parameters": parameters,
        "fit_indices": fit_indices,
        "t_front_ps": float(t_front_ps),
        "t_front_error_ps": float(errors[2]),
        "sigma_ps": float(sigma_ps),
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "signal_at_inflection": float(
            signal_at_inflection
        ),
        "fraction_of_maximum": float(
            fraction_of_maximum
        ),
        "r_squared": float(r_squared),
    }


# ============================================================
# MAXIMUM-SLOPE CHECK
# ============================================================

def maximum_slope_inflection(time_ps, signal):
    """
    Nonparametric estimate of the inflection point.
    """
    smoothed = smooth_signal(signal)

    derivative = np.gradient(
        smoothed,
        time_ps,
    )

    peak_index = int(
        np.argmax(smoothed)
    )

    front_index = int(
        np.argmax(
            derivative[:peak_index + 1]
        )
    )

    return float(
        time_ps[front_index]
    )


# ============================================================
# PLOT EACH PROFILE
# ============================================================

def plot_fit(
    filename,
    distance_um,
    time_ps,
    signal,
    fit_result,
    derivative_time_ps,
):
    fit_indices = fit_result[
        "fit_indices"
    ]

    dense_time = np.linspace(
        time_ps[fit_indices[0]],
        time_ps[fit_indices[-1]],
        500,
    )

    dense_fit = erf_rising_edge(
        dense_time,
        *fit_result["parameters"],
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        time_ps,
        signal,
        label="Data",
    )

    plt.plot(
        dense_time,
        dense_fit,
        linewidth=2,
        label="Error-function fit",
    )

    plt.axvline(
        fit_result["t_front_ps"],
        linestyle="--",
        linewidth=2,
        label=(
            rf"Fit: $t_F="
            rf"{fit_result['t_front_ps']:.1f}"
            rf"\pm"
            rf"{fit_result['t_front_error_ps']:.1f}$ ps"
        ),
    )

    plt.axvline(
        derivative_time_ps,
        linestyle=":",
        linewidth=2,
        label=(
            rf"Maximum slope: "
            rf"{derivative_time_ps:.1f} ps"
        ),
    )

    plt.scatter(
        fit_result["t_front_ps"],
        fit_result[
            "signal_at_inflection"
        ],
        s=60,
        zorder=5,
        label="Inflection point",
    )

    plt.xlabel(
        "Time [ps]",
        fontsize=15,
    )

    plt.ylabel(
        r"$T_r$ [A.U.]",
        fontsize=15,
    )

    plt.title(
        rf"$z={distance_um:g}\,\mu$m",
        fontsize=16,
    )

    plt.tick_params(
        labelsize=13,
    )

    plt.legend(
        fontsize=10,
    )

    plt.tight_layout()

    output_name = SCRIPT_DIR / (
        f"fit_z_{distance_um:g}_um.png"
    )

    plt.savefig(
        output_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    results = []

    for filename in CSV_FILES:
        csv_path = Path(filename)
        if not csv_path.is_absolute():
            csv_path = SCRIPT_DIR / csv_path

        print(
            f"\nAnalyzing {csv_path.name}"
        )

        try:
            distance_um = (
                extract_distance_from_filename(
                    csv_path.name
                )
            )

            time_ps, signal = load_profile(
                csv_path
            )

            fit_result = (
                fit_inflection_point(
                    time_ps,
                    signal,
                )
            )

            derivative_time_ps = (
                maximum_slope_inflection(
                    time_ps,
                    signal,
                )
            )

            method_difference_ps = abs(
                fit_result["t_front_ps"]
                - derivative_time_ps
            )

            combined_uncertainty_ps = np.sqrt(
                fit_result[
                    "t_front_error_ps"
                ]**2
                + method_difference_ps**2
            )

            results.append({
                "filename": filename,
                "filepath": str(csv_path),
                "distance_um": distance_um,
                "t_front_ps": fit_result[
                    "t_front_ps"
                ],
                "fit_error_ps": fit_result[
                    "t_front_error_ps"
                ],
                "maximum_slope_time_ps": (
                    derivative_time_ps
                ),
                "method_difference_ps": (
                    method_difference_ps
                ),
                "combined_uncertainty_ps": (
                    combined_uncertainty_ps
                ),
                "sigma_ps": fit_result[
                    "sigma_ps"
                ],
                "fraction_of_maximum": (
                    fit_result[
                        "fraction_of_maximum"
                    ]
                ),
                "r_squared": fit_result[
                    "r_squared"
                ],
            })

            plot_fit(
                filename=csv_path.name,
                distance_um=distance_um,
                time_ps=time_ps,
                signal=signal,
                fit_result=fit_result,
                derivative_time_ps=(
                    derivative_time_ps
                ),
            )

            print(
                f"z = {distance_um:g} µm\n"
                f"Inflection time = "
                f"{fit_result['t_front_ps']:.2f} "
                f"± {fit_result['t_front_error_ps']:.2f} ps\n"
                f"Maximum-slope time = "
                f"{derivative_time_ps:.2f} ps\n"
                f"Fraction of maximum = "
                f"{fit_result['fraction_of_maximum']:.3f}\n"
                f"R² = "
                f"{fit_result['r_squared']:.5f}"
            )

        except Exception as error:
            print(
                f"Could not analyze "
                f"{csv_path}: {error}"
            )

    results = pd.DataFrame(results)

    if results.empty:
        raise RuntimeError(
            "No profiles were successfully analyzed."
        )

    results = (
        results
        .sort_values("distance_um")
        .reset_index(drop=True)
    )

    results_path = SCRIPT_DIR / RESULTS_FILE

    results.to_csv(
        results_path,
        index=False,
    )

    print("\nFinal results:")
    print(
        results.to_string(
            index=False
        )
    )

    print(
        f"\nSaved results to "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()