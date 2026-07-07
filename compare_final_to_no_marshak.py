from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parameters import (
    Material,
    Experiment,
    t_array_TD,
    T_array_TD,
    alpha,
    beta,
    g,
    f,
    mu,
    rho,
    lambda_param,
    c,
    a_hev,
)
from model_main import analytic_wave_front_no_marshak
from csv_helpers import ensure_dir, save_series_csv


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    for i in range(1, len(y)):
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return out


def derivation_front(
    times_ns: np.ndarray,
    *,
    chi_r: float = 0.0,
    chi_m: float = 0.0,
) -> np.ndarray:
    """
    Plots/uses the derived approximation:
      x_F^2(t) ~= [2 D0 A_R a / (A_m I_M)] * T_s^{-2 beta}(t) * int_0^t T_s^{4+alpha+beta}(t') dt'

    With profiles:
      R(y) = (1-y)(1+chi_r y)   -> A_R = -R'(0) = 1-chi_r
      M^beta(y) = (1-y)(1+chi_m y) -> I_M = int_0^1 M^beta(y) dy = 1/2 + chi_m/6
    """
    t_ns = np.asarray(times_ns, dtype=float)
    t_sec = t_ns * 1e-9

    # Surface temperature from parameters drive profile (eV -> HeV)
    Ts_hev = np.interp(t_ns, t_array_TD, T_array_TD) * 0.01

    D0 = (c * g / 3.0) * rho ** (-1.0 - lambda_param)
    A_m = f * rho ** (1.0 - mu)
    A_R = 1.0 - chi_r
    I_M = 0.5 + chi_m / 6.0

    if A_R <= 0:
        raise ValueError("A_R must be positive. Choose chi_r < 1.")
    if I_M <= 0:
        raise ValueError("I_M must be positive. Choose chi_m > -3.")

    integrand = Ts_hev ** (4.0 + alpha + beta)
    integral = cumulative_trapezoid(integrand, t_sec)

    prefactor = (2.0 * D0 * A_R * a_hev) / (A_m * I_M)
    x_f_sq = prefactor * np.maximum(Ts_hev, 1e-20) ** (-2.0 * beta) * integral
    x_f_cm = np.sqrt(np.maximum(x_f_sq, 0.0))
    return x_f_cm


def save_outputs(
    times_ns: np.ndarray,
    x_derivation_cm: np.ndarray,
    x_no_marshak_cm: np.ndarray,
    chi_r: float,
    chi_m: float,
) -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data_new" / Experiment / Material / "derivation_comparison"
    fig_dir = base_dir / "Figures_new" / Experiment / Material / "derivation_comparison"
    ensure_dir(data_dir)
    ensure_dir(fig_dir)

    csv_path = data_dir / "derivation_vs_no_marshak.csv"
    save_series_csv(
        csv_path,
        {
            "time_ns": times_ns,
            "x_derivation_cm": x_derivation_cm,
            "x_no_marshak_cm": x_no_marshak_cm,
            "delta_derivation_minus_no_marshak_cm": x_derivation_cm - x_no_marshak_cm,
            "ratio_derivation_over_no_marshak": np.divide(
                x_derivation_cm,
                np.maximum(x_no_marshak_cm, 1e-14),
            ),
        },
    )

    plt.figure(figsize=(8, 6))
    plt.plot(times_ns, x_derivation_cm, color="black", linewidth=2, label="Derivation formula")
    plt.plot(times_ns, x_no_marshak_cm, color="green", linestyle="--", linewidth=2, label="No Marshak")
    plt.xlabel(r"$t$ [ns]")
    plt.ylabel(r"$x_F$ [cm]")
    plt.title(rf"Derivation vs No Marshak: $\chi_R={chi_r:.2f},\ \chi_M={chi_m:.2f}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = fig_dir / "derivation_vs_no_marshak.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return csv_path, fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot x_F(t) from the derived closed-form approximation using parameters.py")
    parser.add_argument("--n-points", type=int, default=250, help="Number of sampled times in the drive range")
    parser.add_argument("--chi-r", type=float, default=0.0, help="Curvature parameter in R(y)=(1-y)(1+chi_r y)")
    parser.add_argument("--chi-m", type=float, default=0.0, help="Curvature parameter in M^beta(y)=(1-y)(1+chi_m y)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t_min = float(np.min(t_array_TD))
    t_max = float(np.max(t_array_TD))
    times_ns = np.linspace(t_min, t_max, args.n_points)

    x_derivation_cm = derivation_front(times_ns, chi_r=args.chi_r, chi_m=args.chi_m)
    x_no_marshak_cm = analytic_wave_front_no_marshak(times_ns, use_seconds=False)
    csv_path, fig_path = save_outputs(
        times_ns,
        x_derivation_cm,
        x_no_marshak_cm,
        args.chi_r,
        args.chi_m,
    )

    print(f"Material: {Material} | Experiment: {Experiment}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
