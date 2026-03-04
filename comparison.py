from parameters import *
from model_main import *
from simulation import *
from scipy.interpolate import interp1d
from csv_helpers import *
from plot_helpers import *

DATA_DIR = BASE_DIR / "Data_new" / Experiment / Material
print(f"Data directory: {DATA_DIR}")
FIGURES_OUTPUT_DIR = BASE_DIR / "figures"
# -----------------------------
# Plotting helpers
# -----------------------------
# plot temperature profiles, front positions, total energies for all ts in stored_t - in 3 different functions.

def plot_front_positions(stored_t, front_positions, analytic_positions=None, marshak_boundary=False, energy_lost_to_gold=False):
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plt.plot(stored_t, front_positions, color='black', linestyle="-", label="Simulated Front Position")
    if analytic_positions is not None and not energy_lost_to_gold:
        plt.plot(
            stored_t, analytic_positions,
            linestyle="--",
            label="Analytic x_F(t) (T_s = T_bath(t))" if not marshak_boundary else "Analytic x_F(t) (Marshak BC)"
        )
    if energy_lost_to_gold:
        plt.plot(
            stored_t, analytic_positions,
            linestyle="--",
            label="Analytic x_F(t) with energy lost to Gold wall"
        )

        plot_csv_series(
            article_front_path("gold_wall.csv"),
            y_scale=10,
            linestyle="-.",
            label="article 1 x_F(t) with energy lost to Gold wall",
        )
        plot_csv_series(
            article_front_path("ablation_block.csv"),
            y_scale=10,
            linestyle="-.",
            label="article 2 x_F(t) wall lost ablation block",
        )
        plot_csv_series(
            article_front_path("2D_Heyney.csv"),
            y_scale=10,
            linestyle="-.",
            label="2D_Heyney",
        )

    plt.xlabel("Time (ns)", fontname='serif')
    plt.ylabel("Wave Front Position (cm)", fontname='serif')
    plt.title(f"Wave Front Position vs Time  - Material: {Material}", fontname='serif')
    plt.grid(True)
    plt.legend(prop={'family': 'serif'})
    plt.tight_layout()

    # # annotate the std on the plot, make it look good with a box
    # plt.annotate(f"Std Dev from analytical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
    #                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    save_figure("front_position.png")


def plot_temperature_profiles(stored_t, stored_Tm):
    colors = plt.cm.viridis(np.linspace(0, 1, len(stored_t)))
    plt.figure(figsize=(8, 6))
    for Ti, ti, color in zip(stored_Tm, stored_t, colors):
        plt.plot(z, Ti, label=f"t={ti:.1e} ns", color=color)
    plt.xlabel("z (cm)")
    plt.ylabel(r"$T(z,t)/T_{\mathrm{bath}}$")
    plt.grid(True)
    plt.legend()
    plt.title(f"Temperature profiles over time - Material: {Material}")
    plt.tight_layout()
    save_figure("temperature_profiles.png")

def plot_energies(stored_t, total_energies, marshak_boundary=False, energy_lost_to_gold=False, ablation=False, vary_rho=False):
    # analytical_points = [analytical_total_energy(ti, rho, T_bath_hev  ) for ti in stored_t]
    # stdev_percent = np.mean(np.abs((total_energies - analytical_points) / analytical_points)) * 100
    # print(f"[tau={tau}] Standard deviation from analytical: {stdev_percent:.3e}")
    plt.figure(figsize=(8,6))
    if marshak_boundary:
        _, _, E_2D, E_wall_array, *_ = analytic_wave_front_dispatch(stored_t, use_seconds=True, mode="marshak_wall_loss", vary_rho=False)
        _, _, E_1D, _, *_ = analytic_wave_front_dispatch(stored_t, use_seconds=True, mode="marshak", vary_rho=vary_rho)
        plt.plot(stored_t, E_1D, label="material energy 1D", linestyle="--", color='purple')
        plt.plot(stored_t, E_2D, label="material energy 2D", linestyle="--", color='orange')
        if energy_lost_to_gold:
            plt.plot(stored_t, E_wall_array, label="energy lost to gold wall", linestyle="-", color='black')
    total_energies_hJ = total_energies * 100 *np.pi*R_cm**2  # convert hJ/mm^2 to hJ

    plot_csv_series(
        article_energy_path("gold_wall_flattop.csv"),
        linestyle="-.",
        label="gold_wall_flattop_energy - article 1",
        color='cyan',
    )
    plot_csv_series(
        article_energy_path("total_energy_2D.csv"),
        linestyle="-.",
        label="total_energy_2D - article 1",
        color='green',
    )
    plot_csv_series(
        article_energy_path("total_energy_1D.csv"),
        linestyle="-",
        label="total_energy_1D - article 1",
        color='orange',
    )

    plt.plot(stored_t, total_energies_hJ, color='blue', linestyle="-.", label="Simulated Material Energy")
    plt.xlabel("Time (ns)")
    plt.ylabel("Total Energy (hJ)")
    plt.title(f"Total Energy vs Time - Material: {Material}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # # annotate the std on the plot, make it look good with a box
    # plt.annotate(f"Std Dev from analytical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
    #     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_figure("total_energy.png")


# -----------------------------
# One-call case runner (removes repeated code)
# -----------------------------
def run_case(*,   times_to_store, reset_initial_conditions = True, marshak_boundary=False):
    """
    Runs:
      1) init state
      2) time loop
      3) front/energy extraction
      4) plots
    """
    E, UR = init_state()
    stored_t, stored_Um, stored_Tm, stored_TR = run_time_loop(E, UR, times_to_store=times_to_store, marshak_boundary=marshak_boundary)
    # store values in csv files
    # use pandas to save 2D arrays
    os.makedirs(DATA_DIR, exist_ok=True)
    # sorted_T is a 2D array where each row is a stored_T at a time step. 
    # Create a csv file where each column is a time step and each row is the temperature profile (for positions) at that time step.
    # make sure that this is saved correctly.
    if marshak_boundary:
        df_sorted_Tm_marshak = pd.DataFrame(stored_Tm)
        df_sorted_Tm_marshak.to_csv(DATA_DIR / "stored_Tm_marshak.csv", header=False, index=False)
        df_sorted_TR_marshak = pd.DataFrame(stored_TR)
        df_sorted_TR_marshak.to_csv(DATA_DIR / "stored_TR_marshak.csv", header=False, index=False)
        df_sorted_Um_marshak = pd.DataFrame(stored_Um)
        df_sorted_Um_marshak.to_csv(DATA_DIR / "stored_Um_marshak.csv", header=False, index=False)
        df_stored_t_marshak = pd.DataFrame(stored_t)
        df_stored_t_marshak.to_csv(DATA_DIR / "stored_time_marshak.csv", header=False, index=False)
        return {
            "stored_t_marshak": np.array(stored_t),
            "stored_Tm_marshak": stored_Tm,
            "stored_TR_marshak": stored_TR,
            "stored_Um_marshak": stored_Um,
        }
    else:
        df_sorted_Tm = pd.DataFrame(stored_Tm)
        df_sorted_Tm.to_csv(DATA_DIR / "stored_Tm.csv", header=False, index=False)
        df_sorted_TR = pd.DataFrame(stored_TR)
        df_sorted_TR.to_csv(DATA_DIR / "stored_TR.csv", header=False, index=False)
        df_sorted_Um = pd.DataFrame(stored_Um)
        df_sorted_Um.to_csv(DATA_DIR / "stored_Um.csv", header=False, index=False)
        df_stored_t = pd.DataFrame(stored_t)
        df_stored_t.to_csv(DATA_DIR / "stored_time.csv", header=False, index=False)
        return {
            "stored_t": np.array(stored_t),
            "stored_Tm": stored_Tm,
            "stored_TR": stored_TR,
            "stored_Um": stored_Um,
        }
    
def plot_front_positions_and_energies(show_plots=True, marshak_boundary=False, energy_lost_to_gold=False, ablation=False, vary_rho=False):
    """Reads back stored csv files and plots front positions and total energies."""
    # read back the stored values from csv files using pandas
    if marshak_boundary:
        stored_Tm = pd.read_csv(DATA_DIR / "stored_Tm_marshak.csv", header=None).to_numpy() #convert to numpy array
        stored_TR = pd.read_csv(DATA_DIR / "stored_TR_marshak.csv", header=None).to_numpy() #convert to numpy array
        stored_Um = pd.read_csv(DATA_DIR / "stored_Um_marshak.csv", header=None).to_numpy() #convert to numpy array
        stored_t = pd.read_csv(DATA_DIR / "stored_time_marshak.csv", header=None).to_numpy().flatten() #convert to 1D numpy array
        front_positions, total_energies = compute_front_and_energy(stored_Um, stored_Tm)
    else:
        stored_Tm = pd.read_csv(DATA_DIR / "stored_Tm.csv", header=None).to_numpy() #convert to numpy array
        stored_TR = pd.read_csv(DATA_DIR / "stored_TR.csv", header=None).to_numpy() #convert to numpy array
        stored_Um = pd.read_csv(DATA_DIR / "stored_Um.csv", header=None).to_numpy() #convert to numpy array
        stored_t = pd.read_csv(DATA_DIR / "stored_time.csv", header=None).to_numpy().flatten() #convert to 1D numpy array
        front_positions, total_energies = compute_front_and_energy(stored_Um, stored_Tm)

    # ---- analytic on the same stored times ----
    if energy_lost_to_gold:
        if ablation:
            chosen_mode = "marshak_ablation"
        else:
            chosen_mode = "marshak_wall_loss"
    else:
        chosen_mode = "marshak" if marshak_boundary else "no_marshak"
    if marshak_boundary:
        dispatch_out = analytic_wave_front_dispatch(stored_t, use_seconds=True, mode=chosen_mode, vary_rho=vary_rho)
        analytic_positions = extract_front_positions(dispatch_out)
    else:
        dispatch_out = analytic_wave_front_dispatch(stored_t, use_seconds=True, mode="no_marshak", vary_rho=False)
        analytic_positions = extract_front_positions(dispatch_out)

    ensure_figures_dir()
    plot_temperature_profiles(stored_t, stored_Tm)
    plot_front_positions(stored_t, front_positions, analytic_positions=analytic_positions, marshak_boundary=marshak_boundary, energy_lost_to_gold=energy_lost_to_gold)
    plot_energies(stored_t, total_energies, marshak_boundary=marshak_boundary, energy_lost_to_gold=energy_lost_to_gold, ablation=ablation, vary_rho=False)
    if show_plots:
        plt.show()
    else:
        plt.close('all')

    #return front_positions, total_energies


#I want to plot the surface temperature (at z=0) where there is marshak boundary condition vs when there is not marshak boundary conditiondef plot_surface_temperature_comparison(stored_t_m, stored_Tm_m, stored_t_nm, stored_Tm_nm):
def plot_surface_temperature_comparison(times_to_store):
    E, UR = init_state()
    stored_times, _, stored_Tm_m = run_time_loop(E, UR, times_to_store, marshak_boundary=True)[0:3]
    surface_temps_m = [Tm[0] for Tm in stored_Tm_m]
    E, UR = init_state()
    surface_temps_nm = []
    for t_query in stored_times:
        surface_temps_nm.append(get_TD(t_query, t_array_TD, T_array_TD))
    _, Ts_1D, _, _, *_ = analytic_wave_front_dispatch(stored_times,  use_seconds=True, mode = "marshak", vary_rho=False)
    _, Ts_2D, _, _, *_ = analytic_wave_front_dispatch(stored_times,  use_seconds=True, mode = "marshak_wall_loss", vary_rho=False)
    plt.figure(figsize=(8, 6))
    plt.plot(stored_times, surface_temps_m, label="With Marshak BC", color='blue', linestyle='--')
    plt.plot(stored_times, Ts_1D, label="Analytic Ts(t) (Marshak BC)", color='blue', linestyle='-')
    plt.plot(stored_times, Ts_2D, label="Analytic Ts(t) with Gold wall loss", color='red', linestyle='-')
    plt.plot(stored_times, surface_temps_nm, label="Without Marshak BC", color='green', linestyle='-')

    plot_csv_series(
        article_temperature_path("surface_marshak.csv"),
        y_scale=100,
        linestyle="-.",
        label="article 1 surface temp with marshak boundary condition",
    )
    plot_csv_series(
        article_temperature_path("surface_gold_lost.csv"),
        y_scale=100,
        linestyle="-.",
        label="article 1 surface temp with gold wall loss",
    )

    plt.xlabel("Time (ns)")
    plt.ylabel(r"Surface Temperature $T(z=0,t)$ (HeV)")
    plt.title("Surface Temperature Comparison with and without Marshak Boundary Condition")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    save_figure("surface_temperature_comparison.png", model1_5=True)


def plot_both_marshak_and_nonmarshak_heat_fronts(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, wall_material='Gold', lam_eff_power=1.5)
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_energy_lost_gold = front_series["analytic_positions_gold_loss"]
    analytic_positions_ablation_const_rho = front_series["analytic_positions_ablation_const_rho"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    analytic_positions_no_marshak = front_series["analytic_positions_no_marshak"]
    T_s_array = front_series["Ts_marshak_gold_loss"]
    bessel_data = front_series["bessel_data_gold_loss"]

    analytic_positions = analytic_wave_front_dispatch(times_to_store, use_seconds=True, mode="no_marshak", vary_rho=False)  # stored_t is ns

    plt.figure(figsize=(8, 6))
    plot_standard_front_analytic_models(
        times_to_store, 
        analytic_positions_marshak=analytic_positions_marshak, 
        analytic_positions_gold_loss=analytic_positions_energy_lost_gold,
        analytic_positions_2D_lam_eff=analytic_positions_2D_lam_eff,
        analytic_positions_2D=analytic_positions_2D,
        analytic_positions_ablation_const_rho=analytic_positions_ablation_const_rho,
        analytic_positions_no_marshak=analytic_positions_no_marshak,
        )

    plot_csv_errorbar(article_front_path("exp_results_back.csv"), y_scale=10,xerr=0.1,fmt='o',capsize=4,elinewidth=1.5,markersize=10,label="Experimental data (article 1)", color='black')


    plt.xlabel("Time (ns)", fontsize=18, fontname='serif')
    plt.ylabel("Wave Front Position (cm)", fontsize=18, fontname='serif')
    plt.ylim(0,0.2)
    plt.title(f"Wave Front Position vs Time  - Material: {Material}", fontsize=18, fontname='serif')
    plt.grid(True)
    plt.legend(prop={'family': 'serif'})
    plt.tight_layout()

    # # annotate the std on the plot, make it look good with a box
    # plt.annotate(f"Std Dev from analytical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
    #                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    save_figure("front_position - marshak_vs_nonmarshak.png", model1_5=True)
    export_analytic_positions_csv(
        times_to_store,
        {
            "front_position": {
                "Marshak": analytic_positions_marshak,
                "Ablation with varying rho": analytic_positions_2D,
                "2D effects + lam_eff": analytic_positions_2D_lam_eff,
                "Ablation with const rho": analytic_positions_ablation_const_rho,
                "Gold Loss": analytic_positions_energy_lost_gold,
                "No Marshak": analytic_positions_no_marshak,
            }
        },
        output_csv_path=DATA_DIR / "1.5 model" / "analytic_positions.csv",
    )
    # Plot radial front profiles z_F(r,t) = z_F(t) * J_0(kappa_0 * r)
    if bessel_data and analytic_positions_energy_lost_gold is not None:
        # Plot 2D spatial view showing front in (r,z) geometry
        plot_2D_front_spatial(bessel_data, analytic_positions_energy_lost_gold,
                             times_to_store, times_ns=[1.0, 2.0, 2.5])
        # Plot temperature heatmaps T(r,z,t)
        plot_temperature_heatmap_2D(bessel_data, analytic_positions_energy_lost_gold,
                        T_s_array, times_to_store, times_ns=[1.0, 2.0, 2.5],
                        ablation=True)

def plot_2D_front_spatial(bessel_data, z_F_array, times_array, times_ns=[1.0, 2.0, 2.5]):
    """
    Plot z_F(r,t) in 2D spatial coordinates (r, z) showing the curved front
    from z=0 to z=L at different times.
    
    Parameters
    ----------
    bessel_data : dict
        Dictionary with time (ns) as keys, containing Bessel function data
    z_F_array : array
        Array of front positions z_F(t) in cm
    times_array : array
        Array of times (in ns) corresponding to z_F_array
    times_ns : list
        List of times (in ns) to plot
    """
    # Find closest available times in data
    available_times = np.array(list(bessel_data.keys()))
    
    fig, axes = plt.subplots(1, len(times_ns), figsize=(18, 8))
    if len(times_ns) == 1:
        axes = [axes]
    
    for idx, t_target in enumerate(times_ns):
        # Find closest time in data
        t_closest = available_times[np.argmin(np.abs(available_times - t_target))]
        data = bessel_data[t_closest]
        
        r_grid = data['r_grid']
        z_grid = data['z_grid']
        J0_profiles = data['J0_profiles']
        J0_profiles_approx = data['J0_profiles_approx']
        kappa_0 = data['kappa_0']
        kappa_0_approx = data['kappa_0_approx']
        albedo = data['albedo']
        
        # Get z_F(t) at this time
        z_F_t = np.interp(t_closest, times_array, z_F_array)
        
        # Interpolate J_0 profile at z = z_F(t)
        J0_at_front = np.zeros_like(r_grid)
        J0_approx_at_front = np.zeros_like(r_grid)
        for r_idx in range(len(r_grid)):
            # Interpolate J_0(r) values along z to get value at z_F
            J0_interpolator = interp1d(z_grid, J0_profiles[:, r_idx], 
                                       kind='linear', fill_value='extrapolate')
            J0_at_front[r_idx] = J0_interpolator(z_F_t)
            J0_approx_interpolator = interp1d(z_grid, J0_profiles_approx[:, r_idx],
                                        kind='linear', fill_value='extrapolate')
            J0_approx_at_front[r_idx] = J0_approx_interpolator(z_F_t)
        # Normalize so that z_F(r=0, t) = z_F(t)
        J0_at_center = J0_at_front[0]
        J_approx_at_center = J0_approx_at_front[0]
        if J0_at_center != 0:
            J0_normalized = J0_at_front / J0_at_center
            J0_normalized_approx = J0_approx_at_front / J_approx_at_center
        else:
            J0_normalized = J0_at_front
            J0_normalized_approx = J0_approx_at_front
        
        # Compute radial front profile: z_F(r,t) = z_F(t) * J_0(kappa_0 * r / R)
        z_F_radial = z_F_t * J0_normalized
        z_F_radial_approx = z_F_t * J0_normalized_approx
        
        ax = axes[idx]
        
        # Plot the domain boundaries
        ax.axhline(y=0, color='black', linewidth=2, label='z = 0 (wall)')
        ax.axhline(y=L, color='gray', linewidth=2, linestyle='--', label=f'z = L = {L:.2f} cm')
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.5, label='r = 0 (axis)')
        ax.axvline(x=R_cm, color='gray', linewidth=2, linestyle='--', label=f'r = R = {R_cm:.2f} cm')
        
        # Plot the curved front position
        ax.plot(r_grid, z_F_radial, linewidth=3, color='red', 
                label=f'Front z_F(r,t)', linestyle='-', markersize=4)
        # ax.plot(r_grid, z_F_radial_approx, linewidth=2, color='blue',
        #         label=f'Approx Front z_F(r,t)', linestyle='--', markersize=4)
        if t_target != 2.5:
            df = pd.read_csv(BASE_DIR / "2D" / "data" / f"{int(t_target)}ns.csv")
        else:
            df = pd.read_csv(BASE_DIR / "2D" / "data" / "2.5ns.csv")
        r_csv = df["x"].to_numpy()
        z_csv = df["y"].to_numpy()
        ax.plot(r_csv, z_csv, color='blue', label=f'Simulated front ({t_target:.2f} ns)', linestyle='--')
        
        # Shade the region behind the front (heated region)
        ax.fill_between(r_grid, 0, z_F_radial, alpha=0.3, color='orange', 
                        label='Heated region')
        
        ax.set_xlabel('Radial position r (cm)', fontsize=12, fontname='serif')
        ax.set_ylabel('Axial position z (cm)', fontsize=12, fontname='serif')
        ax.set_title(f't = {t_closest:.2f} ns\nkappa_0 = {kappa_0:.3f}, kappa_0_approx = {kappa_0_approx:.3f}, albedo = {albedo:.3f}', fontsize=12, fontname='serif')
        ax.set_ylim([0, L])
        ax.set_xlim([0, R_cm])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best', prop={'family': 'serif'})
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('2D Spatial View of Heat Front z_F(r,t) in Cylindrical Geometry', fontsize=14, y=1.00, fontname='serif')
    plt.tight_layout()
    
    save_figure("2D_front_spatial.png", model2_D=True, dpi=150, bbox_inches='tight')
    plt.close()


def plot_temperature_heatmap_2D(bessel_data, z_F_array, T_s_array, times_array, times_ns=[1.0, 2.0, 2.5], ablation=False):
    """
    Plot 2D temperature heatmaps T(r,z,t) = T_s(t) * (1 - z/z_F(r,t))^(1/(4+alpha-beta))
    in cylindrical geometry for specified times.
    
    Parameters
    ----------
    bessel_data : dict
        Dictionary with time (ns) as keys, containing Bessel function data
    z_F_array : array
        Array of front positions z_F(t) in cm
    T_s_array : array
        Array of surface temperatures T_s(t) in heV
    times_array : array
        Array of times (in ns) corresponding to z_F_array and T_s_array
    times_ns : list
        List of times (in ns) to plot
    """
    from scipy.interpolate import interp1d
    
    # Get material parameters from global scope
    exponent = 1.0 / (4.0 + alpha - beta)  # Self-similar profile exponent
    
    # Find closest available times in data
    available_times = np.array(list(bessel_data.keys()))
    
    data_of_R = None
    if ablation:
        data_of_R = R_of_t_z(times_to_store=times_array, show_plot=False, verbose=False)

    fig, axes = plt.subplots(1, len(times_ns), figsize=(18, 6))
    if len(times_ns) == 1:
        axes = [axes]
    
    for idx, t_target in enumerate(times_ns):
        # Find closest time in data
        t_closest = available_times[np.argmin(np.abs(available_times - t_target))]
        data = bessel_data[t_closest]
        
        r_grid = data['r_grid']
        z_grid = data['z_grid']
        J0_profiles = data['J0_profiles']
        kappa_0 = data['kappa_0']
        albedo = data['albedo']
        
        # Get z_F(t) and T_s(t) at this time
        z_F_t = np.interp(t_closest, times_array, z_F_array)
        T_s_t = np.interp(t_closest, times_array, T_s_array)
        
        # Create 2D mesh grid for temperature calculation
        n_r = 100
        n_z = 200
        r_mesh = np.linspace(0.0, R_cm, n_r)
        z_mesh = np.linspace(0.0, L, n_z)
        R_mesh, Z_mesh = np.meshgrid(r_mesh, z_mesh)
        
        # Compute z_F(r,t) for each radial position
        z_F_radial = np.zeros(n_r)
        for i_r, r_val in enumerate(r_mesh):
            # Interpolate J_0 at z = z_F_t to get radial variation
            if r_val <= r_grid[-1]:
                # Interpolate J_0 profiles at z = z_F_t
                J0_at_front = np.zeros_like(r_grid)
                for j_r, r_j in enumerate(r_grid):
                    J0_interpolator = interp1d(z_grid, J0_profiles[:, j_r], 
                                               kind='linear', fill_value='extrapolate')
                    J0_at_front[j_r] = J0_interpolator(z_F_t)
                
                # Interpolate to get J_0 at current r_val
                J0_interp_r = interp1d(r_grid, J0_at_front, kind='linear', 
                                       fill_value='extrapolate')
                J0_at_r = J0_interp_r(r_val)
                
                # Normalize by centerline value
                J0_center = J0_at_front[0] if J0_at_front[0] != 0 else 1.0
                z_F_radial[i_r] = z_F_t * (J0_at_r / J0_center)
            else:
                z_F_radial[i_r] = z_F_t
        
        # Compute temperature T(r,z,t) on the mesh
        T_mesh = np.zeros_like(R_mesh)
        for i_z in range(n_z):
            for i_r in range(n_r):
                z_val = z_mesh[i_z]
                z_F_r = z_F_radial[i_r]
                
                if z_val < z_F_r and z_F_r > 0:
                    # Self-similar profile: T = T_s * (1 - z/z_F)^exponent
                    T_mesh[i_z, i_r] = T_s_t * ((1.0 - z_val / z_F_r) ** exponent)
                else:
                    # Beyond front: T = 0
                    T_mesh[i_z, i_r] = 0.0
        
        ax = axes[idx]
        
        # Create heatmap with gouraud shading
        pcm = ax.pcolormesh(R_mesh, Z_mesh, T_mesh, shading='gouraud', cmap='Spectral_r')
        
        # Add contour lines with temperature labels in eV
        # Use evenly spaced levels from 0 to T_s to avoid clustering near front
        contour_levels = np.array([1,1.2,1.3,1.4, 1.5]) # 4 evenly spaced levels
        contour = ax.contour(R_mesh, Z_mesh, T_mesh, levels=contour_levels, colors='black', 
                            linewidths=0.5, alpha=0.3)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # Plot the front position
        ax.plot(r_mesh, z_F_radial, linewidth=3, color='cyan', 
                label=f'Front z_F(r,t)', linestyle='--')

        # Plot ablation contour r = R(t,z) if available
        if data_of_R is not None and len(data_of_R) > 0:
            r_times = np.array(list(data_of_R.keys()), dtype=float)
            t_r = r_times[np.argmin(np.abs(r_times - t_closest))]
            r_profile = np.asarray(data_of_R[t_r], dtype=float)
            z_profile = np.asarray(z, dtype=float)

            if z_profile.size == r_profile.size and z_profile.size > 1:
                r_interp = np.interp(z_mesh, z_profile, r_profile, left=np.nan, right=np.nan)
                valid = np.isfinite(r_interp)
                if np.any(valid):
                    r_interp_plot = np.clip(r_interp[valid], 0.0, R_cm)
                    ax.plot(r_interp_plot, z_mesh[valid], color='white', linewidth=2.5,
                            linestyle='-', label='Ablation contour R(t,z)')
        
        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Temperature T (heV)')
        
        # Add grounded shading at z=0 (ground boundary)
        
        # Domain boundaries
        ax.axhline(y=0, color='white', linewidth=2, linestyle='-', alpha=0.7)
        ax.axhline(y=L, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(x=R_cm, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel('Radial position r (cm)')
        ax.set_ylabel('Axial position z (cm)')
        ax.set_title(f't = {t_closest:.2f} ns\nT_s = {T_s_t:.3f} heV, α = {albedo:.3f}')
        ax.set_ylim([0, L])  # Focus on heated region
        ax.set_xlim([0, R_cm])
        ax.legend(fontsize=9, loc='upper right')
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Temperature Distribution T(r,z,t) with Self-Similar Profile (exponent = {exponent:.3f})', 
                y=1.00, fontsize=20)
    plt.tight_layout()
    
    save_figure("temperature_heatmap_2D.png", model2_D=True, dpi=150, bbox_inches='tight')
    plt.close()


def compare_with_article_2_exp1_Massen(times_to_store):
    stored_Tm_marshak = pd.read_csv(DATA_DIR / "stored_Tm_marshak.csv", header=None).to_numpy() #convert to numpy array
    stored_Um_marshak = pd.read_csv(DATA_DIR / "stored_Um_marshak.csv", header=None).to_numpy() #convert to numpy array
    stored_t_marshak = pd.read_csv(DATA_DIR / "stored_time_marshak.csv", header=None).to_numpy().flatten() #convert to 1D numpy array
    front_positions_marshak, total_energies_marshak = compute_front_and_energy(stored_Um_marshak, stored_Tm_marshak)
    analytic_positions_marshak, Ts,_,_,*_ = analytic_wave_front_dispatch(times_to_store,use_seconds=True,mode="marshak",vary_rho=False)  # stored_t is ns

    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_analytic_if_available(times_to_store, analytic_positions_marshak, label="Analytic x_F(t) (Marshak BC)", linestyle="--", color='green')
    plt.plot(stored_t_marshak, front_positions_marshak, color='red', linestyle="-.", label="Simulated Front Position")

    plot_csv_curves([
        {"path": article_front_path("150.csv"), "y_scale": 10, "linestyle": "-", "label": "HR Pure", "color": "blue"},
        {"path": article_front_path("120.csv"), "y_scale": 10, "linestyle": "-.", "label": "HR Doped", "color": "red"},
        {"path": article_front_path("100.csv"), "y_scale": 10, "linestyle": "-", "label": "1D Analytic Model Pure", "color": "black"},
    ])

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.03)
    plt.title(f"Wave Front Position vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # # annotate the std on the plot, make it look good with a box
    # plt.annotate(f"Std Dev from analytical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
    #                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # make directory if not exists
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
    save_figure("front_position - compare Massen.png", model1_5=True)

    plt.figure(figsize=(8, 6))
    if Ts is not None:
        plt.plot(
            times_to_store, Ts,
            linestyle="--",
            label="Analytic Ts(t) (Marshak BC)",
            color='green'
        )
    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Surface Temperature Ts (HeV)", fontsize = 18)
    plt.title(f"Surface Temperature vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("surface_temperature - compare Massen.png", model1_5=True)

def compare_with_article_2_exp2_Xu(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=2)
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_ablation_varying_rho = front_series["analytic_positions_2D"]
    analytic_positions_no_marshak = front_series["analytic_positions_no_marshak"]
    analytic_positions_ablation_varying_rho_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    bessel_data = front_series["bessel_data_2D"]
    Ts_1D = front_series["Ts_1D"]
    Ts_2D = front_series["Ts_2D"]
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_ablation_varying_rho,
        analytic_positions_no_marshak=analytic_positions_no_marshak,
        analytic_positions_2D_lam_eff=analytic_positions_ablation_varying_rho_lam_eff,
    )

    plot_csv_curves([
        {"path": article_front_path("HR_pure.csv"), "y_scale": 10, "linestyle": "-", "label": "HR Pure", "color": "blue"},
        {"path": article_front_path("HR_doped.csv"), "y_scale": 10, "linestyle": "-.", "label": "HR Doped", "color": "green"},
        {"path": article_front_path("1D_front_pure.csv"), "y_scale": 10, "linestyle": "-", "label": "1D Analytic Model Pure", "color": "black"},
        {"path": article_front_path("2D_front_pure.csv"), "y_scale": 10, "linestyle": "--", "label": "2D Analytic Model Pure", "color": "black"},
        {"path": article_front_path("1D_front_doped.csv"), "y_scale": 10, "linestyle": "-", "label": "1D Analytic Model Doped", "color": "red"},
        {"path": article_front_path("2D_front_doped.csv"), "y_scale": 10, "linestyle": "--", "label": "2D Analytic Model Doped", "color": "red"},
    ])

    plot_csv_errorbars([
        {"path": article_front_path("exp_results_pure.csv"), "y_scale": 10, "xerr": 0.0, "label": "Expt. pure", "color": "black"},
        {"path": article_front_path("exp_results_doped.csv"), "y_scale": 10, "xerr": 0.0, "label": "Expt. doped", "color": "red"},
    ])

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.05)
    plt.xlim(0,1.2)
    plt.title(f"Wave Front Position vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # # annotate the std on the plot, make it look good with a box
    # plt.annotate(f"Std Dev from analytical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
    #                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # make directory if not exists
    save_figure("front_position - compare Xu.png", model1_5=True)

    plt.figure(figsize=(8, 6))
    plot_standard_surface_temperature_models(times_to_store, Ts_1D=Ts_1D, Ts_2D=Ts_2D)

    plot_csv_curves([
        {"path": article_temperature_path("T_drive.csv"), "y_scale": 100, "linestyle": "--", "label": "T_D", "color": "green"},
        {"path": article_temperature_path("Ts_1D_pure.csv"), "y_scale": 100, "linestyle": "-", "label": "Ts 1D model", "color": "blue"},
        {"path": article_temperature_path("Ts_2D_pure.csv"), "y_scale": 100, "linestyle": "--", "label": "Ts 2D model", "color": "black"},
    ])
    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("T (HeV)", fontsize = 18)
    plt.xlim(0.1,2)
    plt.ylim(0,2)
    plt.title(f"Temperature vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_figure("Temperatures - compare Xu.png", model1_5=True)
    export_analytic_positions_csv(
        times_to_store,
        {
            "front_position": {
                "Marshak": analytic_positions_marshak,
                "Ablation with varying rho": analytic_positions_ablation_varying_rho,
                "2D effects + lam_eff": analytic_positions_ablation_varying_rho_lam_eff,
                "No Marshak": analytic_positions_no_marshak,
            }
        },
        output_csv_path=DATA_DIR / "1.5 model" / "analytic_positions.csv",
    )

    if bessel_data and analytic_positions_ablation_varying_rho is not None:
        # Plot 2D spatial view showing front in (r,z) geometry
        plot_2D_front_spatial(bessel_data, analytic_positions_ablation_varying_rho,
                             times_to_store, times_ns=[1.0, 2.0, 2.5])
        # Plot temperature heatmaps T(r,z,t)
        plot_temperature_heatmap_2D(bessel_data, analytic_positions_ablation_varying_rho,
                        Ts_2D, times_to_store, times_ns=[1.0, 2.0, 2.5],
                        ablation=True)


def compare_with_article_2_exp3_13a(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=1)
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    analytic_position_Be_lost, _, _, _, *_ = analytic_wave_front_dispatch(times_to_store,use_seconds=True,mode="marshak_wall_loss",vary_rho=False, wall_material='Be')  # stored_t is ns
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_2D,
        analytic_positions_2D_lam_eff=analytic_positions_2D_lam_eff,
    )
    if analytic_position_Be_lost is not None:
        plt.plot(
            times_to_store, analytic_position_Be_lost,
            linestyle="-",
            label="Analytic x_F(t) (Be Lost)",
            color='cyan'
        )
    plot_csv_errorbars([
        {"path": article_front_path("exp_results_gold.csv"), "y_scale": 10, "xerr": 0.03, "label": "Expt. Gold", "color": "black"},
        {"path": article_front_path("exp_results_be.csv"), "y_scale": 10, "xerr": 0.03, "label": "Expt. Be", "color": "orange"},
    ])

    plot_csv_curves([
        {"path": article_front_path("1D_front_gold.csv"), "y_scale": 10, "linestyle": "--", "label": "T_D", "color": "red"},
        {"path": article_front_path("2D_front_gold.csv"), "y_scale": 10, "linestyle": "-", "label": "Ts 1D model", "color": "black"},
        {"path": article_front_path("2D_front_Be.csv"), "y_scale": 10, "linestyle": "--", "label": "Ts 2D model", "color": "orange"},
    ])

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.15)
    plt.title(f"Wave Front Position vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # # annotate the std on the plot, make it look good with a box
    # plt.annotate(f"Std Dev from analytical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
    #                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # make directory if not exists
    save_figure("front_position - compare Back Ta2O5.png", model1_5=True)

def compare_with_article_2_exp4_14(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=1)
    analytic_positions_no_marshak = front_series["analytic_positions_no_marshak"]
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    power_law = (4 + alpha - beta) / 4
    # analytic_positions_2D = analytic_positions_2D *(1-0.5**power_law) 
    # analytic_positions_2D_lam_eff = analytic_positions_2D_lam_eff *(1-0.5**power_law)
    # analytic_positions_marshak = analytic_positions_marshak *(1-0.5**power_law)
    # analytic_positions_no_marshak = analytic_positions_no_marshak *(1-0.5**power_law)

    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_2D,
        analytic_positions_2D_lam_eff=analytic_positions_2D_lam_eff,
        analytic_positions_no_marshak=analytic_positions_no_marshak,
    )

    plot_csv_curves([
        {"path": article_front_path("HR.csv"), "y_scale": 10, "linestyle": "-.", "label": "HR", "color": "green"},
        {"path": article_front_path("1D_model.csv"), "y_scale": 10, "linestyle": "-", "label": "1D Model", "color": "blue"},
        {"path": article_front_path("2D_model.csv"), "y_scale": 10, "linestyle": "--", "label": "2D Model", "color": "black"},
    ])

    plot_csv_errorbars([
        {"path": article_front_path("exp_results.csv"), "y_scale": 10, "label": "Expt.", "color": "black"},
    ])

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.2)
    plt.title(f"Wave Front Position vs Time  - Material: {Material} (Figure 14)", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("front_position - compare Back SiO2 low energy.png", model1_5=True)
    
def compare_with_article_2_exp5_15a(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=1)
    analytic_position_HR = front_series["analytic_positions_no_marshak"]
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_2D,
        analytic_positions_no_marshak=analytic_position_HR,
        analytic_positions_2D_lam_eff=analytic_positions_2D_lam_eff,
    )

    plot_csv_curves([
        {"path": article_front_path("HR.csv"), "y_scale": 10, "linestyle": "--", "label": "HR", "color": "green"},
        {"path": article_front_path("1D_front.csv"), "y_scale": 10, "linestyle": "-", "label": "1D Analytic Model", "color": "blue"},
        {"path": article_front_path("2D_front.csv"), "y_scale": 10, "linestyle": "-", "label": "2D Analytic Model", "color": "black"},
    ])

    plot_csv_errorbars([
        {"path": article_front_path("exp_results.csv"), "y_scale": 10, "yerr": 0.01, "label": "Expt.", "color": "black"},
    ])

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.25)
    plt.title(f"Wave Front Position vs Time  - Material: {Material} (Figure 15a)", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("front_position - compare Moore SiO2.png", model1_5=True)

def compare_with_article_2_exp5_15b(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=2)
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_ablation_vary_rho = front_series["analytic_positions_2D"]
    analytic_position_HR = front_series["analytic_positions_no_marshak"]
    analytic_positions_ablation_vary_rho_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_ablation_vary_rho,
        analytic_positions_2D_lam_eff=analytic_positions_ablation_vary_rho_lam_eff,
        analytic_positions_no_marshak=analytic_position_HR,
    )

    plot_csv_curves([
        {"path": article_front_path("HR.csv"), "y_scale": 10, "linestyle": "--", "label": "HR", "color": "green"},
        {"path": article_front_path("1D_front.csv"), "y_scale": 10, "linestyle": "-", "label": "1D Analytic Model", "color": "blue"},
        {"path": article_front_path("2D_front.csv"), "y_scale": 10, "linestyle": "-", "label": "2D Analytic Model", "color": "black"},
    ])

    plot_csv_errorbars([
        {"path": article_front_path("exp_results.csv"), "y_scale": 10, "yerr": 0.005, "label": "Expt.", "color": "black"},
    ])

    plt.xlabel("Time (ns)")
    plt.ylabel("Wave Front Position (cm)")
    plt.ylim(0,0.3)
    plt.title(f"Wave Front Position vs Time  - Material: {Material} (Figure 15b)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    save_figure("front_position - compare Moore C8H7Cl.png", model1_5=True)

def compare_with_article_2_exp6_16(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=1)
    analytic_positions_ablation = front_series["analytic_positions_2D"]
    analytic_positions_ablation_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    plt.figure(figsize=(8, 6))
    power_law = (4 + alpha - beta) / 4
    analytic_positions_ablation = analytic_positions_ablation * (1-0.4**power_law)  # From section V part 2 where f = 0.4 (40% of maximum radiative flux)
    analytic_positions_ablation_lam_eff = analytic_positions_ablation_lam_eff * (1-0.4**power_law)  # From section V part 2 where f = 0.4 (40% of maximum radiative flux)

    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_2D=analytic_positions_ablation,
        analytic_positions_2D_lam_eff=analytic_positions_ablation_lam_eff,
    )

    plot_csv_curves([
        {"path": article_front_path("2D_front_pure.csv"), "y_scale": 10, "linestyle": "-", "label": "2D Analytic Model - pure (article)", "color": "red"},
        {"path": article_front_path("2D_front_doped.csv"), "y_scale": 10, "linestyle": "-", "label": "2D Analytic Model - doped (article)", "color": "black"},
    ])

    plot_csv_errorbars([
        {"path": article_front_path("exp_results_pure.csv"), "y_scale": 10, "xerr": 0.01, "yerr": 0.001, "label": "Expt. pure", "color": "red"},
        {"path": article_front_path("exp_results_doped.csv"), "y_scale": 10, "xerr": 0.01, "yerr": 0.001, "label": "Expt. doped", "color": "black"},
    ])
    
    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.1)
    plt.title(f"Wave Front Position vs Time  - Material: {Material} (Figure 16)", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure(f"front_position - compare Keiter ({Material}).png", model1_5=True)

def compare_with_article_2_exp7_17(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, wall_material="Vacuum", lam_eff_power=1)
    analytic_positions_no_marshak = front_series["analytic_positions_no_marshak"]
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    Ts_1D = front_series["Ts_1D"]
    Ts_2D = front_series["Ts_2D"]

    plt.figure(figsize=(8, 6))
    power_law = (4 + alpha - beta) / 4
    analytic_positions_vacuum_lost = analytic_positions_vacuum_lost * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    analytic_positions_marshak = analytic_positions_marshak * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    analytic_positions_no_marshak = analytic_positions_no_marshak * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    analytic_positions_2D = analytic_positions_2D * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    analytic_positions_2D_lam_eff = analytic_positions_2D_lam_eff * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_2D,
        analytic_positions_no_marshak=analytic_positions_no_marshak,
        analytic_positions_2D_lam_eff=analytic_positions_2D_lam_eff,
    )

    plot_csv_curves([
        {"path": article_front_path("HR.csv"), "y_scale": 10, "linestyle": "-.", "label": "HR (article)", "color": "green"},
        {"path": article_front_path("1D_front.csv"), "y_scale": 10, "linestyle": "-", "label": "1D Analytic Model (article)", "color": "blue"},
        {"path": article_front_path("2D_front.csv"), "y_scale": 10, "linestyle": "--", "label": "2D Analytic Model (article)", "color": "black"},
    ])

    df = pd.read_csv(article_front_path("exp_results.csv"))
    # Adjust column names if needed
    t_csv = df["x"].to_numpy()
    x_csv = df["y"].to_numpy()
    yerr = 0.1*x_csv/10
    # Plot - not the general function for errorbars because we want to customize the error bars
    plt.errorbar(
        t_csv, x_csv/10,
        yerr=yerr,
        fmt='o',
        capsize=4,
        elinewidth=1.5,
        markersize=8,
        label="Expt. doped",
        color='black'
    )

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.025)
    plt.xlim(0,1.2)
    plt.title(f"Wave Front Position vs Time  - Material: {Material} (Figure 17)", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    save_figure("front_position - compare Ji-Yan.png", model1_5=True)

    plt.figure(figsize=(8, 6))
    plot_standard_surface_temperature_models(times_to_store, Ts_1D=Ts_1D, Ts_2D=Ts_2D)

    df = pd.read_csv(article_temperature_path("T_drive.csv"))
    # Adjust column names if needed
    t_csv = df["x"].to_numpy()
    x_csv = df["y"].to_numpy()
    plt.plot(t_csv, x_csv/100, linestyle="--", label="T_D", color='green')

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("T (HeV)", fontsize = 18)
    plt.xlim(0.1,2)
    plt.ylim(0,2)
    plt.title(f"Temperature vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_figure("Temperatures - compare Ji-Yan.png", model1_5=True)

def compare_with_french_gold(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=1)
    analytic_positions_ablation_varying_rho_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    analytic_positions_ablation_const_rho = front_series["analytic_positions_ablation_const_rho"]
    analytic_positions_gold_loss = front_series["analytic_positions_gold_loss"]
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_non_marshak = front_series["analytic_positions_no_marshak"]
    analytic_positions_ablation_varying_rho = front_series["analytic_positions_2D"]
    
    plt.figure(figsize=(8, 6))
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_ablation_varying_rho,
        analytic_positions_no_marshak=analytic_positions_non_marshak,
        analytic_positions_gold_loss=analytic_positions_gold_loss,
        analytic_positions_ablation_const_rho=analytic_positions_ablation_const_rho,
        analytic_positions_2D_lam_eff=analytic_positions_ablation_varying_rho_lam_eff,
    )
    
    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylim(0,0.3)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.title(f"Wave Front Position vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("front_position - french_gold.png", model1_5=True)

    export_analytic_positions_csv(
        times_to_store,
        {
            "HR": analytic_positions_non_marshak,
            "marshak": analytic_positions_marshak,
            "gold_loss": analytic_positions_gold_loss,
            "ablation_const_rho": analytic_positions_ablation_const_rho,
            "ablation_varying_rho": analytic_positions_ablation_varying_rho,
            "ablation_varying_rho_lam_eff": analytic_positions_ablation_varying_rho_lam_eff,
        },
        FIGURES_OUTPUT_DIR / "analytic_positions_french_gold.csv",
    )

def compare_with_french_cupper(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, wall_material="Cupper", lam_eff_power=1)
    analytic_positions_ablation_varying_rho_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    analytic_positions_ablation_const_rho = front_series["analytic_positions_ablation_const_rho"]
    analytic_positions_gold_loss = front_series["analytic_positions_gold_loss"]
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_non_marshak = front_series["analytic_positions_no_marshak"]
    analytic_positions_ablation_varying_rho = front_series["analytic_positions_2D"]
    
    plt.figure(figsize=(8, 6))
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_ablation_varying_rho,
        analytic_positions_no_marshak=analytic_positions_non_marshak,
        analytic_positions_gold_loss=analytic_positions_gold_loss,
        analytic_positions_ablation_const_rho=analytic_positions_ablation_const_rho,
        analytic_positions_2D_lam_eff=analytic_positions_ablation_varying_rho_lam_eff,
    )

    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("Wave Front Position (cm)", fontsize = 18)
    plt.ylim(0,0.25)
    plt.title(f"Wave Front Position vs Time  - Material: {Material} (Figure 16)", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("front_position - french_copper.png", model1_5=True)

def R_of_t_z(times_to_store=None, show_plot=True, verbose=True):
    dispatch_out = analytic_wave_front_dispatch(times_to_store, use_seconds=True, mode="marshak_ablation", vary_rho=True)  # stored_t is ns
    data_of_R = dispatch_out[4]
    # plt.figure(figsize=(8, 6))
    # plt.plot(times_to_store, [data_of_R[t][10] for t in data_of_R.keys()], label="Radius R(t)", color='blue')
    # plt.xlabel("Time (ns)")
    # plt.ylabel("Radius R(t)")
    # plt.title(f"Radius vs Time  - Material: {Material}")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    existing_keys = np.array(list(data_of_R.keys()))
    t1 = existing_keys[np.argmin(np.abs(existing_keys - 1.0))]
    t2 = existing_keys[np.argmin(np.abs(existing_keys - 2.0))]
    t2_5 = existing_keys[np.argmin(np.abs(existing_keys - 2.5))]
    if verbose:
        print(f"data_of_R[t1]: {data_of_R[t1]}, at t={t1} ns")
        print(f"data_of_R[t2]: {data_of_R[t2]}, at t={t2} ns")
        print(f"data_of_R[t2_5]: {data_of_R[t2_5]}, at t={t2_5} ns")

    plt.figure(figsize=(8, 6))
    plt.plot(z, data_of_R[t1], label="Radius R(z,t=10)", color='blue')
    df = pd.read_csv(article_radius_path("1.csv"))
    # Adjust column names if needed
    z_csv = df["x"].to_numpy()
    R_csv = df["y"].to_numpy()
    plt.plot(z_csv/10, R_csv/10, linestyle="--", label="1 ns", color='black')

    plt.plot(z, data_of_R[t2], label="Radius R(z,t=11)", color='red')
    df = pd.read_csv(article_radius_path("2.csv"))
    # Adjust column names if needed
    z_csv = df["x"].to_numpy()
    R_csv = df["y"].to_numpy()
    plt.plot(z_csv/10, R_csv/10, linestyle="--", label="2 ns", color='black')

    plt.plot(z, data_of_R[t2_5], label="Radius R(z,t=11.5)", color='blue')
    df = pd.read_csv(article_radius_path("2.5.csv"))
    # Adjust column names if needed
    z_csv = df["x"].to_numpy()
    R_csv = df["y"].to_numpy()
    plt.plot(z_csv/10, R_csv/10, linestyle="--", label="2.5 ns", color='black')

    plt.xlabel("Position z (cm)")
    plt.ylabel("Radius R(z)")
    plt.title(f"Radius vs position  - Material: {Material}")
    plt.grid(True)
    plt.legend()
    save_figure("Radius_high_SiO2.png", model1_5=True)

    return data_of_R

def plot_albedo_z0_vs_time(times_to_store, mode="marshak_ablation", vary_rho=True, lam_eff=True, power=1.5):
    """
    Plot albedo at z=0 (surface/wall) as a function of time.
    Albedo values are taken from bessel_data generated in wall-loss/ablation modes.
    """
    dispatch_out = analytic_wave_front_dispatch(
        times_to_store,
        use_seconds=True,
        mode=mode,
        vary_rho=vary_rho,
        lam_eff=lam_eff,
        power=power,
    )

    if not isinstance(dispatch_out, tuple) or len(dispatch_out) < 6:
        raise ValueError("Selected mode does not return bessel/albedo data. Use a wall-loss or ablation mode.")

    bessel_data = dispatch_out[5]
    if not bessel_data:
        raise ValueError("No bessel_data available to plot albedo.")

    t_ns = np.array(sorted(bessel_data.keys()), dtype=float)
    albedo_z0 = np.array([bessel_data[t]['albedo'] for t in t_ns], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.plot(t_ns, albedo_z0, color='black', linestyle='-', linewidth=2, label='Albedo at z=0')
    plt.xlabel("Time (ns)")
    plt.ylabel("Albedo")
    plt.title(f"Albedo at z=0 vs Time - Material: {Material}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_figure("albedo_z0_vs_time.png", model2_D=True, dpi=150, bbox_inches='tight')

def compare_n_1(times_to_store):
    analytic_positions_ablation_varying_rho_1, Ts_2D_ablation_varyingrho_1, _, _, *_ = analytic_wave_front_dispatch(times_to_store,use_seconds=True,mode="marshak_ablation",vary_rho=True, lam_eff=True, power=1)  # stored_t is ns
    plt.figure(figsize=(8, 6))
    if analytic_positions_ablation_varying_rho_1 is not None:
        plt.plot(
            times_to_store, analytic_positions_ablation_varying_rho_1 ,
            linestyle="-",
            label="Analytic x_F(t) (ablation + Gold Lost + varying rho, power=1, The changing radius at z=0)",
            color='blue'
        )
    df = pd.read_csv(FIGURES_OUTPUT_DIR / "analytic_positions_french_gold.csv")
    # Adjust column names if needed
    t_csv = df["x"].to_numpy()
    x_csv = df["y"].to_numpy()
    plt.plot(t_csv, x_csv, linestyle="--", label="Analytic x_F(t) (ablation + Gold Lost + varying rho, power=1 and average over radius)", color='red')
    plt.xlabel("Time (ns)", fontsize = 18)
    plt.ylabel("T (HeV)", fontsize = 18)
    plt.title(f"Temperature vs Time  - Material: {Material}", fontsize = 18)
    plt.grid(True)
    plt.legend()
    plt.show()

def simulate():
    times_to_store = t_final * np.linspace(0.01**0.3, 1.0, 150) ** (1/0.3)
    times_to_store = t_final * np.linspace(0.01, 1, 150)

    show_plots = False
    marshak_boundary1 = True
    energy_lost_to_gold1 = True
    ablation1 = True
    vary_rho1 = False
    results_tau0 = run_case(times_to_store=times_to_store, reset_initial_conditions=True, marshak_boundary=marshak_boundary1)
    plot_front_positions_and_energies(show_plots=show_plots, marshak_boundary=marshak_boundary1, energy_lost_to_gold=energy_lost_to_gold1, ablation=ablation1, vary_rho=vary_rho1)
    # results_tau0 = run_case(times_to_store=times_to_store, reset_initial_conditions=True, marshak_boundary=False)
    # plot_front_positions_and_energies(show_plots=show_plots, marshak_boundary=False)

if __name__ == "__main__":
    #simulate()
    times_to_store = np.linspace(0.01, 3, 1000)
    #plot_both_marshak_and_nonmarshak_heat_fronts(times_to_store)
    #compare_with_marshak_results()
    #R_of_t_z(times_to_store=times_to_store)
    #compare_with_article_2_exp1_Massen(times_to_store)
    compare_with_article_2_exp2_Xu(times_to_store)
    #compare_with_article_2_exp3_13a(times_to_store)
    #compare_with_article_2_exp4_14(times_to_store)
    #compare_with_article_2_exp5_15a(times_to_store)
    #compare_with_article_2_exp5_15b(times_to_store)
    #compare_with_article_2_exp6_16(times_to_store)
    #compare_with_article_2_exp7_17(times_to_store)
    #compare_with_french_gold(times_to_store)
    #compare_n_1(times_to_store)
    #compare_with_french_cupper(times_to_store)
    # plot_surface_temperature_comparison(times_to_store)
    plot_albedo_z0_vs_time(times_to_store)