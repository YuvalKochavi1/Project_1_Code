"""Plot the time for the foam front to reach 0.75 mm versus foam diameter.

This script scans the eff_lam run folders produced by
simulation_2d_pipeline.run_eff_lam_radius_sweep(...), reads each
front_position_vs_time_r0.csv file, interpolates the first time at which the
foam front reaches 0.75 mm, and plots that time as a function of the foam
diameter.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csv_helpers import ensure_dir

BASE_DIR = PROJECT_ROOT
DATA_ROOT = BASE_DIR / "Data_new" / "Back" / "SiO2" / "2D_simulation" / "eff_lam"
FIGURE_ROOT = BASE_DIR / "Figures_new" / "Back" / "SiO2" / "2D_simulation" / "eff_lam"
TARGET_FRONT_MM = 0.75


def _radius_from_folder_name(folder_name: str) -> float:
    """Parse a folder name like `R_0p01` into a radius in cm."""
    match = re.fullmatch(r"R_(\d+p\d+|\d+)", folder_name)
    if not match:
        raise ValueError(f"Cannot parse radius from folder name: {folder_name}")
    return float(match.group(1).replace("p", "."))


def _load_crossing_time(csv_path: Path, target_front_mm: float = TARGET_FRONT_MM) -> float:
    """Return the first time in ns when the front reaches target_front_mm.

    Uses linear interpolation between the two nearest samples that straddle the
    target front position. If the front never reaches the target, returns NaN.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return float("nan")

    time_ns = np.asarray(data["time_ns"], dtype=float)
    front_mm = np.asarray(data["front_position_mm"], dtype=float)

    valid = np.isfinite(time_ns) & np.isfinite(front_mm)
    time_ns = time_ns[valid]
    front_mm = front_mm[valid]

    if time_ns.size == 0:
        return float("nan")

    reached = np.where(front_mm >= target_front_mm)[0]
    if reached.size == 0:
        return float("nan")

    idx = int(reached[0])
    if idx == 0:
        return float(time_ns[0])

    t0, t1 = float(time_ns[idx - 1]), float(time_ns[idx])
    f0, f1 = float(front_mm[idx - 1]), float(front_mm[idx])
    if f1 == f0:
        return t1

    frac = (target_front_mm - f0) / (f1 - f0)
    return t0 + frac * (t1 - t0)


def gather_radius_sweep_results(data_root: Path = DATA_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Collect diameters and crossing times from the eff_lam sweep folders."""
    rows: list[tuple[float, float]] = []

    for folder in sorted(data_root.glob("R_*/front_vs_time/front_position_vs_time_r0.csv")):
        radius_folder = folder.parent.parent.name
        radius_cm = _radius_from_folder_name(radius_folder)
        diameter_mm = 2.0 * radius_cm * 10.0
        crossing_time_ns = _load_crossing_time(folder, TARGET_FRONT_MM)
        rows.append((diameter_mm, crossing_time_ns))

    if not rows:
        raise FileNotFoundError(f"No sweep CSVs found under {data_root}")

    rows.sort(key=lambda item: item[0])
    diameters_mm = np.array([item[0] for item in rows], dtype=float)
    crossing_times_ns = np.array([item[1] for item in rows], dtype=float)
    return diameters_mm, crossing_times_ns


def plot_front_time_vs_diameter(*, data_root: Path = DATA_ROOT, figure_root: Path = FIGURE_ROOT) -> Path:
    """Create and save the diameter vs. time plot."""
    diameters_mm, crossing_times_ns = gather_radius_sweep_results(data_root)

    ensure_dir(figure_root)
    figure_path = figure_root / "time_to_front_0p75mm_vs_diameter.png"

    plt.figure(figsize=(8, 5))
    plt.plot(diameters_mm, crossing_times_ns, marker="o", linewidth=2.0, color="tab:blue")
    plt.xlabel("Diameter (mm)")
    plt.ylabel("Time for front to reach 0.75 mm (ns)")
    plt.title("Time to foam front at 0.75 mm vs Diameter")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {figure_path}")
    print("Diameter_mm, time_ns")
    for d_mm, t_ns in zip(diameters_mm, crossing_times_ns):
        print(f"{d_mm:.6g}, {t_ns:.6g}")

    return figure_path


def main() -> None:
    plot_front_time_vs_diameter()


if __name__ == "__main__":
    main()
