"""Compare Back SiO2 and Ta2O5 flux curvature at 1 mm, 400 ps after breakout.

This script uses radiation_flux.py functions directly (no duplicated model logic):
- detect_arrival_time
- _numeric_time_keys
- compute_flux_curvature_at_position
- compute_and_plot_T4_heatmap

Run with no arguments:
    python Comparisons/Back/flux_curvature_1mm_400ps.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DETECTOR_MM = 1.0
DELAY_NS = 0.25 # 250 ps
TIMES_NS = np.linspace(0.01, 4.0, 1400)

OUT_DIR = Path(__file__).resolve().parent / "flux_curvature_1mm_400ps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP_SIO2_PATH = PROJECT_ROOT / "Data_new" / "Back" / "SiO2" / "article" / "flux" / "1mm.csv"
EXP_TA2O5_PATH = PROJECT_ROOT / "Data_new" / "Back" / "SiO2" / "article" / "flux" / "Ta2O5_1mm.csv"


def _normalise_to_center(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if x_vals.size == 0 or y_vals.size == 0:
        return y_vals

    idx0 = int(np.argmin(np.abs(x_vals)))
    y0 = float(y_vals[idx0])
    if y0 > 0:
        return y_vals / y0
    return y_vals


def _load_experiment_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Experimental CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"CSV must contain x,y columns: {csv_path}")

    x_mm = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x_mm) & np.isfinite(y_raw)
    x_mm = x_mm[valid]
    y_raw = y_raw[valid]

    return x_mm, _normalise_to_center(x_mm, y_raw)


def _newest_heatmap_file(directory: Path, before: set[Path]) -> Path:
    after = set(directory.glob("T4_spatial_heatmap_*ns.png"))
    created = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if created:
        return created[-1]

    existing = sorted(after, key=lambda p: p.stat().st_mtime)
    if existing:
        return existing[-1]

    raise FileNotFoundError(f"No heatmap file found in {directory}")


def _worker_compute(material: str, profile_out: Path, heatmap_out: Path) -> None:
    """Compute one material in isolated process context."""
    os.environ["PHYSICS_MATERIAL"] = material

    import model_main  # pylint: disable=import-outside-toplevel
    import parameters  # pylint: disable=import-outside-toplevel
    import radiation_flux  # pylint: disable=import-outside-toplevel
    if material == "SiO2":
        result = model_main.analytic_wave_front_dispatch(
            TIMES_NS,
            use_seconds=True,
            mode="marshak_ablation",
            wall_material="Gold",
            vary_rho=True,
            lam_eff=True,
            power=1.2,
        )
    elif material == "Ta2O5":
        result = model_main.analytic_wave_front_dispatch(
            TIMES_NS,
            use_seconds=True,
            mode="marshak_ablation",
            wall_material="Gold",
            vary_rho=True,
            lam_eff=False,
            power=1,
        )

    x_f = np.asarray(result[0], dtype=float)
    ts = np.asarray(result[1], dtype=float)
    bessel_data = result[5] if len(result) > 5 else {}

    if not bessel_data:
        raise RuntimeError(f"No bessel_data returned for {material}.")

    z_cm = DETECTOR_MM / 10.0
    _, t_breakout_ns = radiation_flux.detect_arrival_time(z_cm, TIMES_NS, x_f)
    if t_breakout_ns is None:
        raise RuntimeError(f"Front never reaches z={DETECTOR_MM:.1f} mm for {material}.")

    t_target_ns = float(t_breakout_ns) + DELAY_NS
    available = radiation_flux._numeric_time_keys(bessel_data)
    if available.size == 0:
        raise RuntimeError(f"No numeric bessel_data times for {material}.")

    t_snapshot_ns = float(available[np.argmin(np.abs(available - t_target_ns))])
    snapshot = bessel_data[t_snapshot_ns]

    r_grid_snap = np.asarray(snapshot["r_grid"], dtype=float)
    z_f_radial = np.asarray(snapshot["z_F_radial"], dtype=float)
    ts_t = float(np.interp(t_snapshot_ns, TIMES_NS, ts))

    ds = radiation_flux.compute_flux_curvature_at_position(
        z_cm,
        r_grid_snap,
        z_f_radial,
        ts_t,
        snapshot=snapshot,
        wall_material="Gold",
        ablation=True,
    )

    r_mm = np.asarray(ds["r_grid"], dtype=float) * 10.0
    flux_raw = np.asarray(ds["flux_radial"], dtype=float)
    flux_norm = _normalise_to_center(r_mm, flux_raw)

    np.savez(
        profile_out,
        material=material,
        r_mm=r_mm,
        flux_norm=flux_norm,
        t_breakout_ns=float(t_breakout_ns),
        t_target_ns=float(t_target_ns),
        t_snapshot_ns=float(t_snapshot_ns),
    )

    # Reuse radiation_flux.py heatmap generator exactly.
    figures_heatmap_dir = (
        Path(model_main.BASE_DIR)
        / "Figures_new"
        / str(parameters.Experiment)
        / str(parameters.Material)
        / "2D_shape"
    )
    figures_heatmap_dir.mkdir(parents=True, exist_ok=True)
    before = set(figures_heatmap_dir.glob("T4_spatial_heatmap_*ns.png"))

    radiation_flux.compute_and_plot_T4_heatmap(
        TIMES_NS,
        mode="marshak_ablation",
        wall_material="Gold",
        use_seconds=True,
        vary_rho=True,
        lam_eff=True,
        power=1,
        show_plot=False,
        save_csv=True,
        time_snapshot_ns=t_snapshot_ns,
    )

    generated_heatmap = _newest_heatmap_file(figures_heatmap_dir, before)
    shutil.copy2(generated_heatmap, heatmap_out)


def _run_worker_subprocess(material: str, profile_out: Path, heatmap_out: Path) -> None:
    env = os.environ.copy()
    env["PHYSICS_MATERIAL"] = material
    env["FC_WORKER_MODE"] = "1"
    env["FC_WORKER_MATERIAL"] = material
    env["FC_PROFILE_OUT"] = str(profile_out)
    env["FC_HEATMAP_OUT"] = str(heatmap_out)

    cmd = [sys.executable, str(Path(__file__).resolve())]
    completed = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker failed for {material}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _plot_combined(model_data: dict[str, dict[str, np.ndarray | float | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    style = {
        "SiO2": {"color": "#d62728", "label": "SiO2"},
        "Ta2O5": {"color": "#1f77b4", "label": "Ta2O5"},
    }

    for mat in ["SiO2", "Ta2O5"]:
        cfg = style[mat]
        r_mm = np.asarray(model_data[mat]["r_mm"], dtype=float)
        flux_norm = np.asarray(model_data[mat]["flux_norm"], dtype=float)
        ax.plot(r_mm, flux_norm, color=cfg["color"], linewidth=2.5, label=f"{cfg['label']} model")

    x_sio2, y_sio2 = _load_experiment_curve(EXP_SIO2_PATH)
    x_ta, y_ta = _load_experiment_curve(EXP_TA2O5_PATH)

    ax.plot(x_sio2, y_sio2, linestyle="--", linewidth=2.2, color=style["SiO2"]["color"], label="SiO2 data (1mm.csv)")
    ax.plot(x_ta, y_ta, linestyle="--", linewidth=2.2, color=style["Ta2O5"]["color"], label="Ta2O5 data (Ta2O5_1mm.csv)")

    ax.set_xlabel(r"$r$ [mm]")
    ax.set_ylabel("Flux [normalized]")
    ax.set_xlim(0.0, 0.9)
    ax.set_ylim(bottom=0.0)
    ax.grid(False)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = OUT_DIR / "flux_curvature_1mm_400ps_SiO2_vs_Ta2O5.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_summary_csv(model_data: dict[str, dict[str, np.ndarray | float | str]]) -> Path:
    out_csv = OUT_DIR / "flux_curvature_1mm_400ps_summary.csv"

    rows = []
    for mat in ["SiO2", "Ta2O5"]:
        r_mm = np.asarray(model_data[mat]["r_mm"], dtype=float)
        flux_norm = np.asarray(model_data[mat]["flux_norm"], dtype=float)
        n = len(r_mm)
        rows.append(
            pd.DataFrame(
                {
                    "material": [mat] * n,
                    "r_mm": r_mm,
                    "flux_norm": flux_norm,
                    "t_breakout_ns": [float(model_data[mat]["t_breakout_ns"])] * n,
                    "t_target_ns": [float(model_data[mat]["t_target_ns"])] * n,
                    "t_snapshot_ns": [float(model_data[mat]["t_snapshot_ns"])] * n,
                }
            )
        )

    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    # Internal worker mode (triggered by env), not user arguments.
    if os.getenv("FC_WORKER_MODE") == "1":
        material = os.environ["FC_WORKER_MATERIAL"]
        profile_out = Path(os.environ["FC_PROFILE_OUT"])
        heatmap_out = Path(os.environ["FC_HEATMAP_OUT"])
        _worker_compute(material, profile_out, heatmap_out)
        return

    model_data: dict[str, dict[str, np.ndarray | float | str]] = {}
    for material in ["SiO2", "Ta2O5"]:
        profile_out = OUT_DIR / f"{material}_profile_1mm_400ps.npz"
        heatmap_out = OUT_DIR / f"T4_heatmap_{material}_1mm_400ps_after_breakout.png"

        _run_worker_subprocess(material, profile_out, heatmap_out)

        bundle = np.load(profile_out, allow_pickle=True)
        model_data[material] = {
            "r_mm": np.asarray(bundle["r_mm"], dtype=float),
            "flux_norm": np.asarray(bundle["flux_norm"], dtype=float),
            "t_breakout_ns": float(bundle["t_breakout_ns"]),
            "t_target_ns": float(bundle["t_target_ns"]),
            "t_snapshot_ns": float(bundle["t_snapshot_ns"]),
            "heatmap_path": str(heatmap_out),
        }

        print(
            f"{material}: breakout={model_data[material]['t_breakout_ns']:.4f} ns, "
            f"target={model_data[material]['t_target_ns']:.4f} ns, "
            f"snapshot={model_data[material]['t_snapshot_ns']:.4f} ns"
        )
        print(f"Saved heatmap -> {heatmap_out}")

    combined_plot = _plot_combined(model_data)
    summary_csv = _save_summary_csv(model_data)

    print(f"Saved combined flux-curvature plot -> {combined_plot}")
    print(f"Saved summary CSV -> {summary_csv}")


if __name__ == "__main__":
    main()
