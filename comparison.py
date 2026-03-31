from parameters import *
from model_main import *
from simulation import *
from scipy.interpolate import interp1d
from scipy import special
from csv_helpers import *
from plot_helpers import *
from shape_2D_analytical_model import plot_2D_front_spatial, plot_temperature_heatmap_2D, plot_temperature_heatmap_2D_series_model
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


def Back_SiO2(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, wall_material='Gold', lam_eff_power=1.5)
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_energy_lost_gold = front_series["analytic_positions_gold_loss"]
    analytic_positions_ablation_const_rho = front_series["analytic_positions_ablation_const_rho"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    analytic_positions_no_marshak = front_series["analytic_positions_no_marshak"]
    Ts_2D = front_series["Ts_2D"]
    Ts_ablation_const_rho = front_series["Ts_ablation_const_rho"]
    Ts_gold_loss = front_series["Ts_marshak_gold_loss"]
    Ts_lam_eff = front_series["Ts_2D_lam_eff"]
    Ts_1D = front_series["Ts_1D"]
    bessel_data_2D = front_series["bessel_data_2D"]
    bessel_data_ablation_const_rho = front_series["bessel_data_ablation_const_rho"]
    bessel_data_2D_lam_eff = front_series["bessel_data_2D_lam_eff"]
    bessel_data_marshak = front_series["bessel_data_marshak"]
    bessel_data_gold_loss = front_series["bessel_data_gold_loss"]
    data_of_R = front_series["data_of_R_2D"]

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

    plot_csv_series(
        article_front_path("ablation_block.csv"),
        linestyle="-.",
        label="const ablation from article",
        color='cyan',
    )

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
    # save_figure("front_position - marshak_vs_nonmarshak.png", model1_5=True)
    save_figure("front_position - ablation varying rho (n=1.5).png", model1_5=True)
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
    # Plot 2D spatial view showing front in (r,z) geometry
    # plot_2D_front_spatial(bessel_data_2D, analytic_positions_2D,
    #                      times_to_store, times_ns=[1.0, 2.0, 2.5])
    # Plot temperature heatmaps T(r,z,t)
    plot_temperature_heatmap_2D(bessel_data_2D, analytic_positions_2D,
                    Ts_2D, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=True, title_suffix="(varying rho)")
    plot_temperature_heatmap_2D(bessel_data_2D_lam_eff, analytic_positions_2D_lam_eff,
                    Ts_lam_eff, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=True, title_suffix="(lam_eff)")
    plot_temperature_heatmap_2D(bessel_data_ablation_const_rho, analytic_positions_ablation_const_rho,
                    Ts_ablation_const_rho, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=True, title_suffix="(const rho)")
    plot_temperature_heatmap_2D(bessel_data_gold_loss, analytic_positions_energy_lost_gold,
                    Ts_gold_loss, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=False, title_suffix="(gold wall loss)")
    plot_temperature_heatmap_2D(bessel_data_marshak, analytic_positions_marshak,
                    Ts_1D, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=False, title_suffix="(Marshak BC)")
    


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
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_position_HR = front_series["analytic_positions_no_marshak"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    Ts_2D = front_series["Ts_2D"]
    Ts_lam_eff = front_series["Ts_2D_lam_eff"]
    bessel_data_2D = front_series["bessel_data_2D"]
    bessel_data_2D_lam_eff = front_series["bessel_data_2D_lam_eff"]
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_2D,
        analytic_positions_2D_lam_eff=analytic_positions_2D_lam_eff,
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

    plot_temperature_heatmap_2D(bessel_data_2D, analytic_positions_2D,
                    Ts_2D, times_to_store, times_ns=[1.5, 2.50, 3.72],
                    ablation=True, title_suffix="(varying rho)")
    plot_temperature_heatmap_2D(bessel_data_2D_lam_eff, analytic_positions_2D_lam_eff,
                    Ts_lam_eff, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=True, title_suffix="(lam_eff)")


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

def compare_with_french_copper(times_to_store):
    front_series = compute_standard_analytic_front_series(times_to_store, wall_material="Copper", lam_eff_power=1)
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
    #Back_SiO2(times_to_store)
    #compare_with_marshak_results()
    #R_of_t_z(times_to_store=times_to_store)
    #compare_with_article_2_exp1_Massen(times_to_store)
    #compare_with_article_2_exp2_Xu(times_to_store)
    #compare_with_article_2_exp3_13a(times_to_store)
    #compare_with_article_2_exp4_14(times_to_store)
    #compare_with_article_2_exp5_15a(times_to_store)
    compare_with_article_2_exp5_15b(times_to_store)
    #compare_with_article_2_exp6_16(times_to_store)
    #compare_with_article_2_exp7_17(times_to_store)
    #compare_with_french_gold(times_to_store)
    #compare_n_1(times_to_store)
    #compare_with_french_copper(times_to_store)
    # plot_surface_temperature_comparison(times_to_store)
    plot_albedo_z0_vs_time(times_to_store)