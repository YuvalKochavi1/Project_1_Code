from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0  # Bessel J0
import mpmath as mp
import matplotlib as mpl
from parameters import Experiment, Material
from csv_helpers import *
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_new" / Experiment / Material
print(f"Data directory: {DATA_DIR}")
# -----------------------------
# Plot style
# -----------------------------
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["mathtext.default"] = "regular"

# optional: precision for mpmath
mp.mp.dps = 50  # digits of precision (30–80 is usually fine)

# -----------------------------
# Geometry
# -----------------------------
R = 0.08   # cm
L = 0.3    # cm

# Grid
Nr = 400
r = np.linspace(0, R, Nr)

# Times
times_ns = np.array([1, 2, 2.5])  # ns
times = times_ns * 1e-9          # seconds (as in your original)

# -----------------------------
# Load front positions (article model)
# -----------------------------
t_ns_csv, z_F_cm_csv = read_xy_csv(DATA_DIR / "2D_simulation" / "gold_supersonic_comparison" / "my1_5_model_supersonic.csv")

idx_1ns = np.argmin(np.abs(t_ns_csv - 1))
idx_2ns = np.argmin(np.abs(t_ns_csv - 2))
idx_2_5ns = np.argmin(np.abs(t_ns_csv - 2.5))
print(f"1ns front position from article: {z_F_cm_csv[idx_1ns]:.4f} cm")
print(f"2ns front position from article: {z_F_cm_csv[idx_2ns]:.4f} cm")
print(f"2.5ns front position from article: {z_F_cm_csv[idx_2_5ns]:.4f} cm")

# -----------------------------
# Model params
# -----------------------------
eps = 0.2

kappa0 = np.sqrt(2 * eps) / R
k0 = np.sqrt(eps) / R
k1 = 2*eps / (np.pi * R)
D = 5e6  # effective diffusion coefficient

def c0(t):
    # t is float seconds
    return (1.0 / kappa0) * np.arccosh(1.0 + D * kappa0**2 * t/2)

def c0_xy(t):
    # t is float seconds
    return (1.0 / k0) * np.arccosh(1.0 + D * k0**2 * t / 2)

def c1_xy(t):
    """
    c1(t) using mpmath.polylog.
    NOTE: returns a Python float.
    """
    c0t = c0_xy(t)  # float
    x = mp.e**(-mp.mpf(k1) * mp.mpf(c0t))  # exp(-k1*c0)

    # expression: 4*Li2(e^{-k1 c0}) - Li2(e^{-2 k1 c0})
    expr = 4 * mp.polylog(2, x) - mp.polylog(2, x**2)

    pref = (-1) * mp.mpf(eps) / (mp.mpf(k1) * mp.pi**2)
    val = pref * expr

    # return float for numpy usage
    return float(val)

# -----------------------------
# Plot analytic curves
# -----------------------------
plt.figure(figsize=(8, 6))

for t, t_ns in zip(times, times_ns):
    idx = np.argmin(np.abs(t_ns_csv - t_ns))
    z_F_1_5D = z_F_cm_csv[idx]

    zF_xy = c0_xy(t) * np.cos(k0 * r)# + c1_xy(t) * np.cos(k1 * r)
    zF_rz  = c0(t) * j0(kappa0 * r)

    # zF *= z_F_1_5D / c0(t)  # optional scaling to match r=0
    zF_rz *= z_F_1_5D / c0(t)  # scale rz to match r=0
    zF_rz /= 1.07
    zF_xy *= 2.4

    print(f"c0({t_ns} ns) = {c0(t)+c1_xy(t):.4f} cm, z_F_1_5D = {z_F_1_5D:.4f} cm")

    zF_xy = np.clip(zF_xy, 0.0, L)
    plt.plot(r, zF_xy, label=f"{t_ns:.2f} ns")
    plt.plot(r, zF_rz, linestyle="--", label=f"{t_ns:.2f} ns (rz)")

# -----------------------------
# Overlay simulation curves
# -----------------------------
for filename, label, color in [
    ("1ns.csv", "1ns", "black"),
    ("2ns.csv", "2ns", "red"),
    ("2.5ns.csv", "2.5ns", "blue"),
]:
    x_sim, y_sim = read_xy_csv(DATA_DIR / "2D_shape" / filename)
    plt.plot(x_sim, y_sim, linestyle="--", label=label, color=color)

plt.xlabel("r (cm)")
plt.ylabel("z_F (cm)")
plt.title(r"Analytic Cylindrical Marshak Front $z_F(r,t)$")
plt.xlim(0, R)
plt.ylim(0, L)
plt.legend()
plt.grid()
save_figure("2D_model_initial_code_fronts.png", model2_D=True, dpi=150, bbox_inches="tight")
plt.show()
