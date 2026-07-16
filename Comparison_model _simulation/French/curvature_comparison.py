import sys
import os
import importlib
import numpy as np
import pandas as pd

# Add the grandparent directory (Project_1_Code) to the Python path
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import parameters (base import)
import parameters
from pathlib import Path
from csv_helpers import BASE_DIR, ensure_dir


def _extract_sorted_time_items(data_dict):
    """Return (times_ns_array, original_time_keys) for numeric-like time keys only."""
    time_items = []
    for key in data_dict.keys():
        # Skip booleans explicitly: bool is a subclass of int.
        if isinstance(key, bool):
            continue
        try:
            t_val = float(key)
        except (TypeError, ValueError):
            continue
        time_items.append((t_val, key))

    time_items.sort(key=lambda item: item[0])
    if not time_items:
        return np.array([]), []

    times_ns = np.array([item[0] for item in time_items], dtype=float)
    time_keys = [item[1] for item in time_items]
    return times_ns, time_keys

def get_arrival_times_2d(bessel_data, z_target_cm):
    """
    Given bessel_data dictionary (keys: time in ns, values: snapshot dict),
    finds the arrival time (in ns) at z = z_target_cm for each radial position.
    """
    times_ns, time_keys = _extract_sorted_time_items(bessel_data)
    if len(times_ns) == 0:
        return None, None
        
    first_snap = bessel_data[time_keys[0]]
    r_grid = first_snap['r_grid'] # in cm
    arrival_times = np.zeros_like(r_grid)
    
    # Loop over each radial grid point to find arrival time
    for j in range(len(r_grid)):
        z_F_t = []
        for key in time_keys:
            z_F_t.append(bessel_data[key]['z_F_radial'][j])
        z_F_t = np.array(z_F_t)
        
        # Interpolate t where z_F_t = z_target_cm
        # np.interp expects xp (z_F_t) to be increasing, which it is since the front moves forward.
        if z_target_cm > z_F_t[-1]:
            # Heat front never reached z_target_cm at this radius in simulated time
            arrival_times[j] = np.nan
        elif z_target_cm < z_F_t[0]:
            # Reached at t = 0
            arrival_times[j] = times_ns[0]
        else:
            arrival_times[j] = np.interp(z_target_cm, z_F_t, times_ns)
            
    return r_grid, arrival_times

def plot_material_albedo(material_name, bessel_data, wall_material):
    """
    Generate and save albedo graphs for the given material:
      1. Surface Albedo (z=0) vs Time
      2. Spatial Albedo Profile vs Depth (z) at 1.0, 2.0, and 3.0 ns
    """
    if not bessel_data:
        print("Warning: bessel_data is empty, cannot plot albedo")
        return
        
    times_ns, time_keys = _extract_sorted_time_items(bessel_data)
    if len(times_ns) == 0:
        return
        
    fig_dir = BASE_DIR / "Figures_new" / "French" / material_name
    ensure_dir(fig_dir)
    
    material_label = material_name.replace('_', r'\_')

    # 1. Surface Albedo vs Time
    surface_albedos = [bessel_data[key]['albedo'] for key in time_keys]
    
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.family': 'serif'})
    plt.plot(times_ns, surface_albedos, color='darkred', linewidth=2.5, label=fr'{wall_material} Surface Albedo ($z=0$)')
    plt.xlabel(r"Time $t$ [ns]", fontsize=13, fontname='serif')
    plt.ylabel(r"Albedo $\alpha$", fontsize=13, fontname='serif')
    plt.title(f"Surface Albedo vs Time\n({material_label} with " + r"$\lambda_{\mathrm{eff}}$" + ")", fontsize=14, fontname='serif', pad=15)
    plt.grid(False)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 4.0)
    plt.legend(prop={'family': 'serif'}, fontsize=11, loc='best')
    plt.tight_layout()
    
    plt.savefig(fig_dir / "albedo_vs_time.png", dpi=300, bbox_inches='tight')
    print(f"Saved surface albedo vs time plot to: {fig_dir / 'albedo_vs_time.png'}")
    plt.close()
    
    # 2. Spatial Albedo Profile vs Depth (z)
    target_times = [1.0, 2.0, 3.0]
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.family': 'serif'})
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for idx, t_target in enumerate(target_times):
        t_closest = min(times_ns, key=lambda k: abs(k - t_target))
        snapshot = bessel_data[t_closest]
        
        z_grid = snapshot['z_grid']  # in cm
        albedo_array = snapshot['albedo_array']
        avg_albedo = snapshot['avg_albedo']
        
        plt.plot(z_grid * 10.0, albedo_array, color=colors[idx], linewidth=2.2, 
                 label=f't = {t_closest:.2f} ns (Avg: {avg_albedo:.3f})')
                 
    plt.xlabel(r"Depth $z$ [mm]", fontsize=13, fontname='serif')
    plt.ylabel(r"Albedo $\alpha$", fontsize=13, fontname='serif')
    plt.title(f"Albedo Profile vs Depth\n({material_label} with " + r"$\lambda_{\mathrm{eff}}$" + ")", fontsize=14, fontname='serif', pad=15)
    plt.grid(False, which='both', linestyle=':', alpha=0.5)
    plt.ylim(0.0, 1.0)
    plt.legend(prop={'family': 'serif'}, fontsize=11, loc='best')
    plt.tight_layout()
    
    plt.savefig(fig_dir / "albedo_profile.png", dpi=300, bbox_inches='tight')
    print(f"Saved albedo profile vs depth plot to: {fig_dir / 'albedo_profile.png'}")
    plt.close()

def run_material_comparison(material_name, z_detector_mm, csv_data_filename, wall_material):
    print("====================================================================")
    print(f"Running Wavefront Arrival Time Comparison for {material_name} at z = {z_detector_mm} mm")
    print("====================================================================")
    
    # Safely reload parameters for the target material
    os.environ["PHYSICS_MATERIAL"] = material_name
    importlib.reload(parameters)
    
    # Reload model_main and all other dependent modules to bind the new parameters
    import eigen_bessel_solver
    import wavefront_helpers
    import wall_loss_model
    import ablation_model
    import albedo_model
    import analytical_wavefront_solver
    import model_main
    
    importlib.reload(eigen_bessel_solver)
    importlib.reload(wavefront_helpers)
    importlib.reload(wall_loss_model)
    importlib.reload(ablation_model)
    importlib.reload(albedo_model)
    importlib.reload(analytical_wavefront_solver)
    importlib.reload(model_main)
    
    z_detector_cm = z_detector_mm / 10.0
    times_to_store = np.linspace(0.01, 4.0, 1000)
    
    # Set the power dynamically based on the material
    power_val = 1 if wall_material == "Gold" else 3
    print(f"Simulating 2D Ablation + lam_eff model (power={power_val})...")
    
    result_ablation_lam_eff = model_main.analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        mode="marshak_ablation",
        vary_rho=True,
        lam_eff=True,
        power=power_val,
        wall_material=wall_material,
    )
    bessel_ablation_lam_eff = result_ablation_lam_eff[5]
    r_grid, t_arrival_ablation_lam_eff = get_arrival_times_2d(bessel_ablation_lam_eff, z_detector_cm)
    
    # Plot Albedo Graphs for this material under the lam_eff model
    print("Plotting albedo graphs...")
    plot_material_albedo(material_name, bessel_ablation_lam_eff, wall_material)
    
    # Load experimental data
    csv_data_path = BASE_DIR / "Data_new" / "French" / material_name / "article" / "front" / csv_data_filename
    print(f"Loading experimental data from: {csv_data_path}")
    if not csv_data_path.exists():
        raise FileNotFoundError(f"Could not find {csv_data_filename} at {csv_data_path}")
    
    data_df = pd.read_csv(csv_data_path)
    
    # Create symmetric grid and curves
    r_symmetric = np.concatenate((-r_grid[::-1], r_grid[1:]))
    t_arr_ablation_lam_eff_sym = np.concatenate((t_arrival_ablation_lam_eff[::-1], t_arrival_ablation_lam_eff[1:]))
    
    # Plotting comparison
    material_label = material_name.replace('_', r'\_')
    plt.figure(figsize=(18, 7))
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'axes.unicode_minus': False,
    })
    
    # Plot model
    plt.plot(r_symmetric, t_arr_ablation_lam_eff_sym, color='purple', linestyle='-', linewidth=2.0, label=r'Model')
    
    # Plot experimental data
    plt.scatter(data_df['x'], data_df['y'], color='black', label=f'Experiment ({csv_data_filename})', marker='o', s=45, zorder=5)
    
    # Customize the plot
    plt.xlabel(r"Radial Location $r$ [cm]", fontsize=22, fontname='serif')
    plt.ylabel(r"Arrival Time $t$ [ns]", fontsize=22, fontname='serif')
    plt.title(f"Wavefront Arrival Time vs Radial Location\nat z = {z_detector_mm} mm ({material_label})", fontsize=24, fontname='serif', pad=18)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Use reloaded parameters R_cm for limit boundary and label
    r_limit = parameters.R_cm
    plt.xlim(-r_limit - 0.01, r_limit + 0.01)
    
    plt.axvline(x=r_limit, color='gray', linestyle='--', alpha=0.7, label=fr'Foam-{wall_material} Interface ($R = {r_limit}$ cm)')
    plt.axvline(x=-r_limit, color='gray', linestyle='--', alpha=0.7)
    
    plt.grid(False, which='both', linestyle=':', alpha=0.5)
    plt.legend(
        prop={'family': 'serif', 'size': 17},
        loc='best',
        markerscale=1.6,
        handlelength=2.0,
        borderpad=0.4,
        labelspacing=0.35,
    )
    plt.tight_layout()
    
    # Ensure directories exist and save plot
    fig_dir = BASE_DIR / "Figures_new" / "French" / material_name
    fig_save_path = fig_dir / f"curvature_comparison_{z_detector_mm}mm.png"
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {fig_save_path}")
    plt.close()
    
    # Export calculated values to CSV for record keeping
    export_df = pd.DataFrame({
        'r_cm': r_grid,
        't_arrival_ablation_lam_eff_ns': t_arrival_ablation_lam_eff
    })
    
    csv_dir = BASE_DIR / "Data_new" / "French" / material_name / "2D_shape"
    ensure_dir(csv_dir)
    csv_save_path = csv_dir / f"model_arrival_times_{z_detector_mm}mm.csv"
    export_df.to_csv(csv_save_path, index=False)
    print(f"Saved calculated arrival times to: {csv_save_path}")
    
    return {
        'r_symmetric': r_symmetric,
        't_arr_ablation_lam_eff_sym': t_arr_ablation_lam_eff_sym,
        'exp_x': data_df['x'].to_numpy(),
        'exp_y': data_df['y'].to_numpy()
    }

def plot_combined_comparison(gold_results, copper_results):
    plt.figure(figsize=(5, 8))
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'axes.unicode_minus': False,
    })
    
    # Plot Gold model & experiment
    plt.plot(gold_results['r_symmetric'], 0.47+gold_results['t_arr_ablation_lam_eff_sym'], color='red', linestyle='-', linewidth=2.2, label=r'Gold Model')
    plt.scatter(gold_results['exp_x'], gold_results['exp_y'], color='red', marker='o', s=45, label='Gold Experiment', zorder=5)
    
    # Plot Copper model & experiment
    plt.plot(copper_results['r_symmetric'], 0.165+copper_results['t_arr_ablation_lam_eff_sym'], color='blue', linestyle='-', linewidth=2.2, label=r'Copper Model')
    plt.scatter(-copper_results['exp_x'], copper_results['exp_y'], color='blue', marker='o', s=45, label='Copper Experiment', zorder=5)
    
    # Interface boundaries
    # plt.axvline(x=0.05, color='red', linestyle=':', alpha=0.5, label='Foam-Gold Interface (R = 0.05 cm)')
    # plt.axvline(x=-0.05, color='red', linestyle=':')
    # plt.axvline(x=0.1, color='blue', linestyle=':', alpha=0.5, label='Foam-Copper Interface (R = 0.1 cm)')
    # plt.axvline(x=-0.1, color='blue', linestyle=':')
    
    plt.xlabel(r"$r$ [cm]", fontsize=22, fontname='serif')
    plt.ylabel(r"Arrival Time $t$ [ns]", fontsize=22, fontname='serif')
    plt.tick_params(axis='both', which='major', labelsize=20)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(spine.get_linewidth() * 2.0)
    
    plt.xlim(0, 0.11)
    ax.grid(False)
    ax.xaxis.grid(False, which='both')
    ax.yaxis.grid(False, which='both')
    plt.legend(
        prop={'family': 'serif', 'size': 19},
        loc='best',
        markerscale=1.3,
        handlelength=2.0,
        borderpad=0.4,
        labelspacing=0.35,
    )
    plt.tight_layout()
    #do axis equal
    fig_dir = BASE_DIR / "Figures_new" / "French"
    ensure_dir(fig_dir)
    fig_save_path = fig_dir / "combined_curvature_comparison.png"
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison plot to: {fig_save_path}")
    plt.close()
    
    try:
        import shutil
        dest_dir = Path("c:/Users/TLP-001/Documents/GitHub/Project_1_docs/presentation/2D_model")
        if dest_dir.exists():
            shutil.copy(fig_save_path, dest_dir / "combined_curvature_comparison.png")
            print(f"Copied combined comparison plot to presentation folder: {dest_dir / 'combined_curvature_comparison.png'}")
    except Exception as e:
        print(f"Could not copy combined plot to presentation: {e}")

def main():
    # 1. Run comparison for SiO2_gold at z = 1.2 mm
    gold_results = run_material_comparison(
        material_name="SiO2_gold",
        z_detector_mm=1.2,
        csv_data_filename="shot1.csv",
        wall_material="Gold"
    )
    
    # 2. Run comparison for SiO2_copper at z = 2.0 mm
    copper_results = run_material_comparison(
        material_name="SiO2_copper",
        z_detector_mm=2.0,
        csv_data_filename="shot2.csv",
        wall_material="Copper"
    )
    
    # 3. Plot both in the same graph
    plot_combined_comparison(gold_results, copper_results)
    
    print("\nAll comparisons executed successfully!")


if __name__ == "__main__":
    main()
