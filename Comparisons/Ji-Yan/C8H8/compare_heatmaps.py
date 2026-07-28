import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.special as special
import matplotlib.pyplot as plt

# 1. Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variable for C8H8
os.environ["PHYSICS_MATERIAL"] = "C8H8"

import parameters as parameters
import model_main as model_main
import importlib
importlib.reload(parameters)
importlib.reload(model_main)

# 2. Get parameters
R_cm = parameters.R_cm  # 0.01 cm
L = parameters.L        # 0.03 cm
K_per_Hev = parameters.K_per_Hev  # 1.1605e6
alpha = parameters.alpha
beta = parameters.beta

# Exponent for temperature profile: T ~ (1 - z/z_F)^(1/(4+alpha-beta))
exponent = 1.0 / (4.0 + alpha - beta)

# 3. Load simulation data
sim_csv_path = SCRIPT_DIR.parents[2] / "Data_new" / "Ji-Yan" / "C8H8" / "2D_simulation" / "temperature_maps_simple" / "heatmap_1.0ns_vacuum.csv"
if not sim_csv_path.exists():
    raise FileNotFoundError(f"Simulation heatmap CSV not found at: {sim_csv_path}")

print(f"Loading simulation data from {sim_csv_path}...")
df_sim = pd.read_csv(sim_csv_path)

# Extract and convert temperature from Kelvin to HeV
z_sim_arr = df_sim["z_cm"].to_numpy()
r_sim_arr = df_sim["r_cm"].to_numpy()
# Note: T_cell_HeV in simulation CSV is saved in Kelvin, convert to heV
T_sim_hev = df_sim["T_cell_HeV"].to_numpy() / K_per_Hev

# Reconstruct 2D simulation grid
z_unique = np.unique(z_sim_arr)
r_unique = np.unique(r_sim_arr)
nz = len(z_unique)
nr = len(r_unique)

z_to_idx = {val: i for i, val in enumerate(z_unique)}
r_to_idx = {val: i for i, val in enumerate(r_unique)}

T_sim_2d = np.zeros((nz, nr))
for idx in range(len(z_sim_arr)):
    i = z_to_idx[z_sim_arr[idx]]
    j = r_to_idx[r_sim_arr[idx]]
    T_sim_2d[i, j] = T_sim_hev[idx]

# 4. Run model and get 1.0 ns snapshot
print("Running 1.5D analytical model...")
times_to_store = np.linspace(0.01, 1.5, 1000)  # in ns
dispatch_out = model_main.analytic_wave_front_dispatch(
    times_to_store,
    use_seconds=True,
    mode="marshak_wall_loss",
    vary_rho=False,
    wall_material="Vacuum",
    lam_eff=False
)
bessel_data = dispatch_out[5]

# Find model snapshot closest to 1.0 ns
closest_t_ns = min(bessel_data.keys(), key=lambda k: abs(k - 1.0))
snapshot = bessel_data[closest_t_ns]
print(f"Loaded model snapshot at: {closest_t_ns:.4f} ns")

kappa_0 = snapshot["kappa_0"]
# Get the model surface temperature and on-axis front position at 1.0 ns
t_idx = np.argmin(np.abs(times_to_store - closest_t_ns))
Ts_1ns = dispatch_out[1][t_idx]
xF_1ns = dispatch_out[0][t_idx]

# 5. Evaluate model at simulation grid points
T_model_2d = np.zeros((nz, nr))
for i, z_val in enumerate(z_unique):
    for j, r_val in enumerate(r_unique):
        if r_val <= R_cm:
            # Model front position at this radius
            z_F = xF_1ns * special.j0(kappa_0 * r_val)
            if z_val < z_F:
                T_model_2d[i, j] = Ts_1ns * ((1.0 - z_val / z_F) ** exponent)
            else:
                T_model_2d[i, j] = 0.0
        else:
            T_model_2d[i, j] = 0.0

# Set figure formatting parameters
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'axes.unicode_minus': False,
})

# 6. Generate side-by-side comparison plot
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

# Common limits and styling
vmin = 0.0
vmax = max(T_sim_2d.max(), T_model_2d.max())
cmap = "Spectral_r"

# Plot Simulation
im1 = axes[0].pcolormesh(r_unique * 1e4, z_unique * 1e4, T_sim_2d, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
axes[0].set_title("Simulation (1.0 ns)", fontsize=14)
axes[0].set_xlabel("r [$\\mu$m]", fontsize=12)
axes[0].set_ylabel("z [$\\mu$m]", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(R_cm * 1e4, color="black", linestyle="--", alpha=0.7)

# Plot Model
im2 = axes[1].pcolormesh(r_unique * 1e4, z_unique * 1e4, T_model_2d, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
axes[1].set_title("Model (1.0 ns)", fontsize=14)
axes[1].set_xlabel("r [$\\mu$m]", fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(R_cm * 1e4, color="black", linestyle="--", alpha=0.7)

# Adjust colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
fig.colorbar(im2, cax=cbar_ax, label="Temperature [heV]")

side_by_side_path = SCRIPT_DIR / "heatmap_comparison_side_by_side.png"
plt.savefig(side_by_side_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved side-by-side comparison to {side_by_side_path}")

# 7. Generate combined/overlay contour plot
plt.figure(figsize=(7, 6))

# Define contour levels in heV
levels = np.linspace(0.1, vmax * 0.95, 5)

# Plot simulation contours
CS_sim = plt.contour(r_unique * 1e4, z_unique * 1e4, T_sim_2d, levels=levels, colors="blue", linestyles="--", linewidths=1.8)
# Plot model contours
CS_model = plt.contour(r_unique * 1e4, z_unique * 1e4, T_model_2d, levels=levels, colors="red", linestyles="-", linewidths=1.8)

# Add custom legend using proxy lines
line_sim = plt.Line2D([0], [0], color="blue", linestyle="--", linewidth=1.8, label="Simulation")
line_model = plt.Line2D([0], [0], color="red", linestyle="-", linewidth=1.8, label="Model")
plt.legend(handles=[line_sim, line_model], loc="best", fontsize=11)

plt.xlabel("r [$\\mu$m]", fontsize=12)
plt.ylabel("z [$\\mu$m]", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axvline(R_cm * 1e4, color="black", linestyle="--", alpha=0.7)

overlay_path = SCRIPT_DIR / "heatmap_comparison_contours.png"
plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved contour overlay comparison to {overlay_path}")
