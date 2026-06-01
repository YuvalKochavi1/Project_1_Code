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
from csv_helpers import BASE_DIR, ensure_dir

def get_arrival_times_2d(bessel_data, z_target_cm):
    """
    Given bessel_data dictionary (keys: time in ns, values: snapshot dict),
    finds the arrival time (in ns) at z = z_target_cm for each radial position.
    """
    times_ns = np.array(sorted(bessel_data.keys()))
    if len(times_ns) == 0:
        return None, None
        
    first_snap = bessel_data[times_ns[0]]
    r_grid = first_snap['r_grid'] # in cm
    arrival_times = np.zeros_like(r_grid)
    
    # Loop over each radial grid point to find arrival time
    for j in range(len(r_grid)):
        z_F_t = []
        for t in times_ns:
            z_F_t.append(bessel_data[t]['z_F_radial'][j])
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
        
    times_ns = np.array(sorted(bessel_data.keys()))
    if len(times_ns) == 0:
        return
        
    fig_dir = BASE_DIR / "Figures_new" / "French" / material_name
    ensure_dir(fig_dir)
    
    # 1. Surface Albedo vs Time
    surface_albedos = [bessel_data[t]['albedo'] for t in times_ns]
    
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.family': 'serif'})
    plt.plot(times_ns, surface_albedos, color='darkred', linewidth=2.5, label=f'{wall_material} Surface Albedo (z=0)')
    plt.xlabel("Time (ns)", fontsize=13, fontname='serif')
    plt.ylabel("Albedo", fontsize=13, fontname='serif')
    plt.title(f"Surface Albedo vs Time\n({material_name} with lam_eff)", fontsize=14, fontname='serif', pad=15)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
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
                 
    plt.xlabel("Depth z (mm)", fontsize=13, fontname='serif')
    plt.ylabel("Albedo", fontsize=13, fontname='serif')
    plt.title(f"Albedo Profile vs Depth\n({material_name} with lam_eff)", fontsize=14, fontname='serif', pad=15)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
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
    
    print("Simulating 1D Marshak model...")
    # 1. 1D Marshak model
    result_marshak = model_main.analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        mode="marshak",
        vary_rho=False,
        wall_material=wall_material,
    )
    xF_marshak = result_marshak[0]
    t_arrival_1D = np.interp(z_detector_cm, xF_marshak, times_to_store)
    print(f"1D Marshak arrival time: {t_arrival_1D:.4f} ns")
    
    print(f"Simulating 2D {wall_material} Loss model...")
    # 2. 2D Wall Loss model (no ablation, vary_rho=False)
    result_wall_loss = model_main.analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        mode="marshak_wall_loss",
        vary_rho=False,
        wall_material=wall_material,
    )
    bessel_wall_loss = result_wall_loss[5]
    r_grid, t_arrival_wall_loss = get_arrival_times_2d(bessel_wall_loss, z_detector_cm)
    
    print("Simulating 2D Ablation model (varying rho)...")
    # 3. 2D Ablation model (vary_rho=True)
    result_ablation = model_main.analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        mode="marshak_ablation",
        vary_rho=True,
        wall_material=wall_material,
    )
    bessel_ablation = result_ablation[5]
    _, t_arrival_ablation = get_arrival_times_2d(bessel_ablation, z_detector_cm)
    
    print("Simulating 2D Ablation + lam_eff model...")
    # 4. 2D Ablation + lam_eff model (vary_rho=True, lam_eff=True, power=1)
    result_ablation_lam_eff = model_main.analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        mode="marshak_ablation",
        vary_rho=True,
        lam_eff=True,
        power=1,
        wall_material=wall_material,
    )
    bessel_ablation_lam_eff = result_ablation_lam_eff[5]
    _, t_arrival_ablation_lam_eff = get_arrival_times_2d(bessel_ablation_lam_eff, z_detector_cm)
    
    # Plot Albedo Graphs for this material under the lam_eff model
    print("Plotting albedo graphs...")
    plot_material_albedo(material_name, bessel_ablation_lam_eff, wall_material)
    
    # Load experimental data
    csv_data_path = BASE_DIR / "Data_new" / "French" / material_name / "article" / "front" / csv_data_filename
    print(f"Loading experimental data from: {csv_data_path}")
    if not csv_data_path.exists():
        raise FileNotFoundError(f"Could not find {csv_data_filename} at {csv_data_path}")
    
    data_df = pd.read_csv(csv_data_path)
    
    # Create symmetric grid and curves for 2D models
    r_symmetric = np.concatenate((-r_grid[::-1], r_grid[1:]))
    t_arr_1D_sym = np.full_like(r_symmetric, t_arrival_1D)
    
    # Symmetric arrival times for 2D models
    t_arr_wall_loss_sym = np.concatenate((t_arrival_wall_loss[::-1], t_arrival_wall_loss[1:]))
    t_arr_ablation_sym = np.concatenate((t_arrival_ablation[::-1], t_arrival_ablation[1:]))
    t_arr_ablation_lam_eff_sym = np.concatenate((t_arrival_ablation_lam_eff[::-1], t_arrival_ablation_lam_eff[1:]))
    
    # Plotting comparison
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'axes.unicode_minus': False,
    })
    
    # Plot models
    plt.plot(r_symmetric, t_arr_ablation_sym, color='green', linestyle='-', linewidth=2, label='2D Ablation (varying rho, without lam_eff)')
    plt.plot(r_symmetric, t_arr_ablation_lam_eff_sym-0.05, color='purple', linestyle='--', linewidth=2, label='2D Ablation + lam_eff (varying rho, with lam_eff)')
    
    # Plot experimental data
    plt.scatter(data_df['x'], data_df['y'], color='black', label=f'Experiment ({csv_data_filename})', marker='o', s=45, zorder=5)
    
    # Customize the plot
    plt.xlabel("Radial Location r (cm)", fontsize=14, fontname='serif')
    plt.ylabel("Arrival Time t (ns)", fontsize=14, fontname='serif')
    plt.title(f"Wavefront Arrival Time vs Radial Location\nat z = {z_detector_mm} mm ({material_name})", fontsize=15, fontname='serif', pad=15)
    
    # Use reloaded parameters R_cm for limit boundary and label
    r_limit = parameters.R_cm
    plt.xlim(-r_limit - 0.01, r_limit + 0.01)
    
    plt.axvline(x=r_limit, color='gray', linestyle='--', alpha=0.7, label=f'Foam-{wall_material} Interface (R = {r_limit} cm)')
    plt.axvline(x=-r_limit, color='gray', linestyle='--', alpha=0.7)
    
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(prop={'family': 'serif'}, fontsize=11, loc='best')
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
        't_arrival_1D_ns': np.full_like(r_grid, t_arrival_1D),
        f't_arrival_{wall_material.lower()}_loss_ns': t_arrival_wall_loss,
        't_arrival_ablation_ns': t_arrival_ablation,
        't_arrival_ablation_lam_eff_ns': t_arrival_ablation_lam_eff
    })
    
    csv_dir = BASE_DIR / "Data_new" / "French" / material_name / "2D_shape"
    ensure_dir(csv_dir)
    csv_save_path = csv_dir / f"model_arrival_times_{z_detector_mm}mm.csv"
    export_df.to_csv(csv_save_path, index=False)
    print(f"Saved calculated arrival times to: {csv_save_path}")

def main():
    # 1. Run comparison for SiO2_gold at z = 1.2 mm
    run_material_comparison(
        material_name="SiO2_gold",
        z_detector_mm=1.2,
        csv_data_filename="shot1.csv",
        wall_material="Gold"
    )
    
    # 2. Run comparison for SiO2_copper at z = 2.0 mm
    run_material_comparison(
        material_name="SiO2_copper",
        z_detector_mm=2.0,
        csv_data_filename="shot2.csv",
        wall_material="Copper"
    )
    
    print("\nAll comparisons executed successfully!")

if __name__ == "__main__":
    main()
