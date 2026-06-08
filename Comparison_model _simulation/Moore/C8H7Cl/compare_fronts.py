import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Set material environment variable and parameters context
os.environ["PHYSICS_MATERIAL"] = "C8H7Cl"
import parameters
parameters.Material = "C8H7Cl"
parameters.Experiment = "Moore"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_helpers import compute_standard_analytic_front_series
from shape_2D_analytical_model import _closest_time_data

# Run standard analytic front series solver for C8H7Cl
times_to_store = np.linspace(0.01, 4, 1000)
print("Running analytical wavefront solver...")
front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=2)

bessel_data_2D = front_series["bessel_data_2D"]
bessel_data_2D_lam_eff = front_series["bessel_data_2D_lam_eff"]

# Extract model front shapes at target time 3.72 ns
t_target = 3.72
t_closest_2D, snapshot_2D = _closest_time_data(bessel_data_2D, t_target)
t_closest_lam, snapshot_lam = _closest_time_data(bessel_data_2D_lam_eff, t_target)

r_model_2D = snapshot_2D['r_grid']
z_F_model_2D = snapshot_2D['z_F_radial'] # in cm

r_model_lam = snapshot_lam['r_grid']
z_F_model_lam = snapshot_lam['z_F_radial'] # in cm

# Load article simulation data
script_dir = Path(__file__).resolve().parent
article_csv_path = script_dir / 'article_simulation.csv'
if not article_csv_path.exists():
    raise FileNotFoundError(f"Could not find article simulation CSV at: {article_csv_path}")

df_article = pd.read_csv(article_csv_path)
r_article = df_article['x'].to_numpy()  # in cm
z_article = df_article['y'].to_numpy() / 10.0 # convert mm to cm

# Load moore front experimental data
moore_csv_path = script_dir / 'moore_front.csv'
if not moore_csv_path.exists():
    moore_csv_path = script_dir.parent / 'moore_front.csv'

if not moore_csv_path.exists():
    raise FileNotFoundError(f"Could not find moore_front.csv at: {moore_csv_path}")

df_moore = pd.read_csv(moore_csv_path)
r_moore = df_moore['r_mm'].to_numpy() / 10.0  # convert mm to cm
z_moore = df_moore['z_mm'].to_numpy() / 10.0  # convert mm to cm

# Sort experimental arrays by increasing radius (needed for np.interp)
sort_idx = np.argsort(r_moore)
r_moore_sorted = r_moore[sort_idx]
z_moore_sorted = z_moore[sort_idx]

# Plot comparison
plt.figure(figsize=(4, 6))
plt.plot(r_moore, z_moore * 10.0, label='Simulation', color='blue', linestyle='--', linewidth=1.2)
plt.plot(r_model_2D, z_F_model_2D * 10.0 + 0.21, label=f'Model', color='red', linewidth=1.2)


plt.xlabel(r'$r$ [cm]', fontsize=12)
plt.ylabel(r'$z_F$ [mm]', fontsize=12)
plt.xlim(0, 0.1)
plt.ylim(2.4, 3.1)
plt.gca().set_aspect(0.1, adjustable='box')
plt.title("Wave front",fontsize=15)


plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()


output_plot_path = script_dir / 'front_comparison_3.72ns.png'
plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
print(f"Saved comparison plot to: {output_plot_path}")

# Print metrics
z_model_2D_interp = np.interp(r_article, r_model_2D, z_F_model_2D)
z_model_lam_interp = np.interp(r_article, r_model_lam, z_F_model_lam)
z_moore_interp = np.interp(r_article, r_moore_sorted, z_moore_sorted)

rmse_2D = np.sqrt(np.mean((z_article - z_model_2D_interp)**2)) * 10.0 # in mm
rmse_lam = np.sqrt(np.mean((z_article - z_model_lam_interp)**2)) * 10.0 # in mm
rmse_2D_exp = np.sqrt(np.mean((z_moore_interp - z_model_2D_interp)**2)) * 10.0 # in mm
rmse_lam_exp = np.sqrt(np.mean((z_moore_interp - z_model_lam_interp)**2)) * 10.0 # in mm

print("\nComparison Summary (values in mm):")
print(f"{'Radius (cm)':<12} | {'Article Sim (mm)':<16} | {'Experiment (mm)':<15} | {'Model 2D (mm)':<13} | {'Model + lam (mm)':<16}")
print("-" * 80)
for i in range(0, len(r_article), max(1, len(r_article)//8)):
    print(f"{r_article[i]:<12.4f} | {z_article[i]*10.0:<16.4f} | {z_moore_interp[i]*10.0:<15.4f} | {z_model_2D_interp[i]*10.0:<13.4f} | {z_model_lam_interp[i]*10.0:<16.4f}")
print("-" * 80)
print(f"RMSE (Model 2D vs Article Sim): {rmse_2D:.4f} mm")
print(f"RMSE (Model 2D + lam vs Article Sim): {rmse_lam:.4f} mm")
print(f"RMSE (Model 2D vs Experiment): {rmse_2D_exp:.4f} mm")

# Also save comparison data to a CSV
comparison_df = pd.DataFrame({
    'r_cm': r_article,
    'z_article_sim_mm': z_article * 10.0,
    'z_experiment_mm': z_moore_interp * 10.0,
    'z_model_2D_mm': z_model_2D_interp * 10.0,
})
output_csv_path = script_dir / 'front_comparison_data_3.72ns.csv'
comparison_df.to_csv(output_csv_path, index=False)
print(f"Saved comparison data to: {output_csv_path}")

