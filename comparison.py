from parameters import *
from model_main import *
from csv_helpers import *
from plot_helpers import *
from shape_2D_analytical_model import plot_2D_front_spatial, plot_temperature_heatmap_2D, plot_temperature_heatmap_2D_series_model
DATA_DIR = BASE_DIR / "Data_new" / Experiment / Material
print(f"Data directory: {DATA_DIR}")
FIGURES_OUTPUT_DIR = BASE_DIR / "Figures_new" / Experiment / Material 

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
    E_gold_loss = front_series["E_gold_loss"]
    E_marshak = front_series["E_marshak"]
    E_wall_gold_loss = front_series["E_W_gold_loss"]
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
    
    # Plot energies
    plt.figure(figsize=(8, 6))
    plt.plot(times_to_store, E_marshak, label="E - Marshak BC", linestyle="-", color='blue')
    plt.plot(times_to_store, E_gold_loss, label="E - Gold Loss", linestyle="-", color='green')
    plt.plot(times_to_store, E_wall_gold_loss, label="E - Gold Wall Loss", linestyle="-", color='red')
    
    #add the energy from the article as a series
    plot_csv_series(
        article_energy_path("gold_wall_flattop.csv"),
        linestyle="-.",
        label="gold wall energy - article 1",
        color='red',
    )
    plot_csv_series(
        article_energy_path("total_energy_2D.csv"),
        linestyle="-.",
        label="total_energy_2D - article 1",
        color='green',
    )
    plot_csv_series(
        article_energy_path("total_energy_1D.csv"),
        linestyle="-.",
        label="total_energy_1D - article 1",
        color='blue',
    )
    #save the simulated energy as a csv series
    if Flattop_condition:
        output_csv_path = DATA_DIR / "2D_shape" / f"simulated_energy_vs_time_{Material}_flattop.csv"
    else:
        output_csv_path = DATA_DIR / "2D_shape" / f"simulated_energy_vs_time_{Material}.csv"
    ensure_dir(DATA_DIR / "energy_comparison")
    save_series_csv(DATA_DIR / "2D_shape" / output_csv_path, {
        "time_ns": np.asarray(times_to_store),
        "E_marshak": E_marshak,
        "E_gold_loss": E_gold_loss,
        "E_wall_gold_loss": E_wall_gold_loss,
        # "E_Be_loss": E_out,
        # "E_Be_wall_loss": Ew_be_out,
    })
    
    plt.xlabel("Time (ns)", fontsize=18, fontname='serif')
    plt.ylabel("Total Energy (hJ)", fontsize=18, fontname='serif')
    plt.title(f"Total Energy vs Time  - Material: {Material}", fontsize=18, fontname='serif')
    plt.grid(True)
    plt.legend(prop={'family': 'serif'})
    plt.tight_layout()
    save_figure("total_energy - Back_SiO2.png", model1_5=True)
    
    #save the analytic positions to csv
    if Flattop_condition:
        output_csv_path = DATA_DIR / "1.5 model" / "analytic_positions_flattop.csv"
    else:
        output_csv_path = DATA_DIR / "1.5 model" / "analytic_positions.csv"
    export_analytic_positions_csv(
        times_to_store,
        {
            "front_position": {
                "Marshak": analytic_positions_marshak,
                "Ablation with varying rho": analytic_positions_2D,
                "2D effects + lam_eff": analytic_positions_2D_lam_eff,
                "Ablation with const rho": analytic_positions_ablation_const_rho,
                "gold loss": analytic_positions_energy_lost_gold,
                "No Marshak": analytic_positions_no_marshak,
                #"Be Loss": analytic_position_Be_lost,
            }
        },
        output_csv_path = output_csv_path,

    )
    
    # Extract z_grid from one of the bessel data snapshots for albedo plots
    z_grid_for_plots = None
    if bessel_data_2D and len(bessel_data_2D) > 0:
        first_snapshot = list(bessel_data_2D.values())[0]
        if 'z_grid' in first_snapshot:
            z_grid_for_plots = first_snapshot['z_grid']
    
    # Plot albedo arrays for different models (only for specific times)
    plot_albedo_arrays(bessel_data_2D, z_grid=z_grid_for_plots, 
                       title="Albedo Profiles - 2D (varying rho)", times_ns=[1.0, 2.0, 2.5])
    save_figure("albedo_profiles_2D_varying_rho.png", model1_5=True)
    
    plot_albedo_arrays(bessel_data_2D_lam_eff, z_grid=z_grid_for_plots,
                       title="Albedo Profiles - 2D (lam_eff)", times_ns=[1.0, 2.0, 2.5])
    save_figure("albedo_profiles_2D_lam_eff.png", model1_5=True)
    
    plot_albedo_arrays(bessel_data_ablation_const_rho, z_grid=z_grid_for_plots,
                       title="Albedo Profiles - Ablation (const rho)", times_ns=[1.0, 2.0, 2.5])
    save_figure("albedo_profiles_ablation_const_rho.png", model1_5=True)
    
    plot_albedo_arrays(bessel_data_gold_loss, z_grid=z_grid_for_plots,
                       title="Albedo Profiles - Gold Loss", times_ns=[1.0, 2.0, 2.5])
    save_figure("albedo_profiles_gold_loss.png", model1_5=True)
    
    plot_albedo_arrays(bessel_data_marshak, z_grid=z_grid_for_plots,
                       title="Albedo Profiles - Marshak BC", times_ns=[1.0, 2.0, 2.5])
    save_figure("albedo_profiles_marshak.png", model1_5=True)
    
    # Plot 2D spatial view showing front in (r,z) geometry
    # plot_2D_front_spatial(bessel_data_2D, analytic_positions_2D,
    #                      times_to_store, times_ns=[1.0, 2.0, 2.5])
    # Plot temperature heatmaps T(r,z,t)
    plot_temperature_heatmap_2D(bessel_data_2D, analytic_positions_2D,
                    Ts_1D, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=True, title_suffix="(varying rho)", color_option = "prr_back")
    plot_temperature_heatmap_2D(bessel_data_2D_lam_eff, analytic_positions_2D_lam_eff,
                    Ts_1D, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=True, title_suffix="(lam_eff)", color_option = "prr_back")
    plot_temperature_heatmap_2D(bessel_data_ablation_const_rho, analytic_positions_ablation_const_rho,
                    Ts_1D, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=True, title_suffix="(const rho)", color_option = "prr_back")
    plot_temperature_heatmap_2D(bessel_data_gold_loss, analytic_positions_energy_lost_gold,
                    Ts_1D, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=False, title_suffix="(gold wall loss)", color_option = "default", flattop=Flattop_condition)
    # plot_temperature_heatmap_2D(bessel_data_marshak, analytic_positions_marshak,
    #                 Ts_1D, times_to_store, times_ns=[1.0, 2.0, 2.5],
    #                 ablation=False, title_suffix="(Marshak BC)", color_option = "prr_back")



def compare_with_article_2_exp1_Massen(times_to_store):
    stored_Tm_marshak = pd.read_csv(DATA_DIR / "stored_Tm_marshak.csv", header=None).to_numpy() #convert to numpy array
    stored_Um_marshak = pd.read_csv(DATA_DIR / "stored_Um_marshak.csv", header=None).to_numpy() #convert to numpy array
    stored_t_marshak = pd.read_csv(DATA_DIR / "stored_time_marshak.csv", header=None).to_numpy().flatten() #convert to 1D numpy array
    analytic_positions_marshak, Ts,_,_,*_ = analytic_wave_front_dispatch(times_to_store,use_seconds=True,mode="marshak",vary_rho=False)  # stored_t is ns

    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_analytic_if_available(times_to_store, analytic_positions_marshak, label="Analytic x_F(t) (Marshak BC)", linestyle="--", color='green')

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
    Ts_marshak = front_series["Ts_1D"]
    E_marshak = front_series["E_marshak"]
    analytic_positions_gold_loss = front_series["analytic_positions_gold_loss"]
    E_2D = front_series["E_2D"]
    E_gold_loss = front_series["E_gold_loss"]
    E_wall_gold_loss = front_series["E_W_gold_loss"]
    E_wall_out_2D = front_series["E_wall_out_2D"]
    bessel_data_2D = front_series["bessel_data_2D"]
    bessel_data_gold_loss = front_series["bessel_data_gold_loss"]
    analytic_position_Be_lost, Ts_out, E_out, Ew_be_out, data_of_R, Be_bessel_data = analytic_wave_front_dispatch(times_to_store,use_seconds=True,mode="marshak_wall_loss",vary_rho=False, wall_material='Be', lam_eff = True, power=1)  # stored_t is ns
    plt.figure(figsize=(8, 6))
    # fit data to analytical
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        # analytic_positions_2D=analytic_positions_2D,
        # analytic_positions_2D_lam_eff=analytic_positions_2D_lam_eff,
        analytic_positions_gold_loss=analytic_positions_gold_loss,
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

    # Plot energies
    plt.figure(figsize=(8, 6))
    plt.plot(times_to_store, E_marshak, label="E - Marshak BC", linestyle="-", color='blue')
    plt.plot(times_to_store, E_out, label="E - Foam with Be Loss", linestyle="-", color='green')
    plt.plot(times_to_store, Ew_be_out, label="E - Be Wall Loss", linestyle="-", color='red')
    
    plt.xlabel("Time (ns)", fontsize=18, fontname='serif')
    plt.ylabel("Total Energy (hJ)", fontsize=18, fontname='serif')
    plt.title(f"Total Energy vs Time  - Material: {Material}", fontsize=18, fontname='serif')
    plt.grid(True)
    plt.legend(prop={'family': 'serif'})
    plt.tight_layout()
    save_figure("total_energy - Ta2O5 Be.png", model1_5=True)

    # Plot energies
    plt.figure(figsize=(8, 6))
    plt.plot(times_to_store, E_marshak, label="E - Marshak BC", linestyle="-", color='blue')
    plt.plot(times_to_store, E_2D, label="E - Foam with Be Loss", linestyle="-", color='green')
    plt.plot(times_to_store, E_wall_out_2D, label="E - 2D Wall Loss", linestyle="-", color='orange')
    
    plt.xlabel("Time (ns)", fontsize=18, fontname='serif')
    plt.ylabel("Total Energy (hJ)", fontsize=18, fontname='serif')
    plt.title(f"Total Energy vs Time  - Material: {Material}", fontsize=18, fontname='serif')
    plt.grid(True)
    plt.legend(prop={'family': 'serif'})
    plt.tight_layout()
    save_figure("total_energy - Ta2O5 Gold.png", model1_5=True)

    if Flattop_condition:
        output_csv_path = DATA_DIR / "1.5 model" / f"simulated_energy_vs_time_{Material}_flattop.csv"
    else:
        output_csv_path = DATA_DIR / "1.5 model" / f"simulated_energy_vs_time_{Material}.csv"
    save_series_csv(output_csv_path, {
        "time_ns": np.asarray(times_to_store),
        "E_marshak": E_marshak,
        "E_Gold_loss": E_gold_loss,
        "E_Gold_wall_loss": E_wall_gold_loss,
        "E_Be_loss": E_out,
        "E_Be_wall_loss": Ew_be_out,
    })

    #save the analytic positions to csv
    if Flattop_condition:
        output_csv_path = DATA_DIR / "1.5 model" / "analytic_positions_flattop.csv"
    else:
        output_csv_path = DATA_DIR / "1.5 model" / "analytic_positions.csv"
    export_analytic_positions_csv(
        times_to_store,
        {
            "front_position": {
                "Marshak": analytic_positions_marshak,
                "gold loss": analytic_positions_gold_loss,
                "Be Loss": analytic_position_Be_lost,
            }
        },
        output_csv_path= output_csv_path,
    )
    z_grid_for_plots = None
    if Be_bessel_data and len(Be_bessel_data) > 0:
        first_snapshot = list(Be_bessel_data.values())[0]
        if 'z_grid' in first_snapshot:
            z_grid_for_plots = first_snapshot['z_grid']
    
    # Plot albedo arrays for different models (only for specific times)
    plot_albedo_arrays(Be_bessel_data, z_grid=z_grid_for_plots, 
                       title="Albedo Profiles - Be lost", times_ns=[0.5, 1.0, 2])
    save_figure("albedo_profiles_Be_Coated.png", model1_5=True)

    plot_temperature_heatmap_2D(Be_bessel_data, analytic_position_Be_lost,
                    Ts_marshak, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=False, title_suffix="(Be wall loss)", color_option = "default", 
                    show_shock=False, wall="Be", flattop=Flattop_condition)
    plot_temperature_heatmap_2D(bessel_data_gold_loss, analytic_positions_gold_loss,
                    Ts_marshak, times_to_store, times_ns=[1.0, 2.0, 2.5],
                    ablation=False, title_suffix="(gold wall loss)", color_option = "default", 
                    show_shock=False, wall="Gold", flattop=Flattop_condition)

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
    """Moore SiO2 experiment"""
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=1)
    analytic_position_HR = front_series["analytic_positions_no_marshak"]
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    Ts_2D = front_series["Ts_2D"]
    Ts_marshak = front_series["Ts_1D"]
    Ts_lam_eff = front_series["Ts_2D_lam_eff"]
    bessel_data_2D = front_series["bessel_data_2D"]
    bessel_data_2D_lam_eff = front_series["bessel_data_2D_lam_eff"]
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

    plot_temperature_heatmap_2D(bessel_data_2D, analytic_positions_2D,
                    Ts_marshak, times_to_store, times_ns=[1, 2, 3],
                    ablation=True, title_suffix="(varying rho)", color_option = "prr_moore", flattop=True)
    plot_temperature_heatmap_2D(bessel_data_2D_lam_eff, analytic_positions_2D_lam_eff,
                    Ts_marshak, times_to_store, times_ns=[1, 2, 3],
                    ablation=True, title_suffix="(lam_eff)", color_option = "prr_moore", flattop=True)


def compare_with_article_2_exp5_15b(times_to_store):
    """Moore C8H7Cl experiment"""
    front_series = compute_standard_analytic_front_series(times_to_store, lam_eff_power=2)
    analytic_positions_marshak = front_series["analytic_positions_marshak"]
    analytic_positions_2D = front_series["analytic_positions_2D"]
    analytic_position_HR = front_series["analytic_positions_no_marshak"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    Ts_2D = front_series["Ts_2D"]
    Ts_marshak = front_series["Ts_1D"]
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

    # Plot surface temperatures
    plt.figure(figsize=(8, 6))
    plot_csv_series(
        article_temperature_path("T_drive.csv"),
        y_scale=100,
        linestyle="--",
        label="T_D (Drive)",
        color="green",
    )
    plot_analytic_if_available(times_to_store, Ts_marshak, label="Marshak (1D)", linestyle="-", color='blue')
    plot_analytic_if_available(times_to_store, Ts_2D, label="2D Model (varying rho)", linestyle="-", color='black')
    plot_analytic_if_available(times_to_store, Ts_lam_eff, label="2D Model (lam_eff)", linestyle="--", color='red')
    plt.xlabel("Time (ns)", fontsize=18)
    plt.ylabel("Surface Temperature T_s (HeV)", fontsize=18)
    plt.title(f"Surface Temperature vs Time  - Material: {Material} (Figure 15b)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    save_figure("temperatures - compare Moore C8H7Cl.png", model1_5=True)

    plot_temperature_heatmap_2D(bessel_data_2D, analytic_positions_2D,
                    Ts_marshak, times_to_store, times_ns=[1.5, 2.50, 3.72],
                    ablation=True, title_suffix="(varying rho)", color_option = "paper", show_shock=False)
    plot_temperature_heatmap_2D(bessel_data_2D_lam_eff, analytic_positions_2D_lam_eff,
                    Ts_marshak, times_to_store, times_ns=[1.0, 2.0, 3.72],
                    ablation=True, title_suffix="(lam_eff)", color_option = "visit")


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
    analytic_positions_gold = front_series["analytic_positions_gold_loss"]
    analytic_positions_2D_lam_eff = front_series["analytic_positions_2D_lam_eff"]
    Ts_1D = front_series["Ts_1D"]
    Ts_2D = front_series["Ts_2D"]
    E_marshak = front_series["E_marshak"]
    E_vacuum_loss = front_series["E_gold_loss"]
    bessel_data_2D = front_series["bessel_data_gold_loss"]

    plt.figure(figsize = (8, 6))
    # power_law = (4 + alpha - beta) / 4
    # analytic_positions_vacuum_lost = analytic_positions_vacuum_lost * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    # analytic_positions_marshak = analytic_positions_marshak * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    # analytic_positions_no_marshak = analytic_positions_no_marshak * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    # analytic_positions_2D = analytic_positions_2D * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    # analytic_positions_2D_lam_eff = analytic_positions_2D_lam_eff * (1 - 0.5**power_law)  # From section V part 2 where f = 0.5 (50% of maximum radiative flux)
    
    plot_standard_front_analytic_models(
        times_to_store,
        analytic_positions_marshak=analytic_positions_marshak,
        analytic_positions_2D=analytic_positions_gold,
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

    #export energies to csv
    save_series_csv(DATA_DIR / "2D_shape" / f"simulated_energy_vs_time_{Material}_vacuum_loss.csv", {
        "time_ns": np.asarray(times_to_store) ,
        "E_marshak": E_marshak,
        "E_vacuum_loss": E_vacuum_loss,
    })

    save_figure("front_position - compare Ji-Yan.png", model1_5=True)
    #save the analytic positions to a csv for later use
    export_analytic_positions_csv(
        times_to_store,
        {
            "front_position": {
                "Marshak": analytic_positions_marshak,
                "Vacuum Loss": analytic_positions_gold,
            }
        },
        DATA_DIR / "1.5 model" / "analytic_positions.csv",
    )

    #export energys to csv
    save_series_csv(DATA_DIR / "2D_shape" / f"simulated_energy_vs_time_{Material}_vacuum_loss.csv", {
        "time_ns": np.asarray(times_to_store) ,
        "E_marshak": E_marshak,
        "E_vacuum_loss": E_vacuum_loss,
    })

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

    plot_temperature_heatmap_2D(bessel_data_2D, analytic_positions_gold,
                    Ts_1D, times_to_store, times_ns=[0.5, 1, 1.3],
                    ablation=False, title_suffix="(vacuum)", flattop=False)
    

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
        DATA_DIR / "1.5 model" / "analytic_positions_french_gold.csv",
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

def plot_albedo_z0_vs_time(times_to_store, mode="marshak_ablation", vary_rho=True, lam_eff=True, power=1.5, wall_material="Gold"):
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
        wall_material=wall_material
    )

    if not isinstance(dispatch_out, tuple) or len(dispatch_out) < 6:
        raise ValueError("Selected mode does not return bessel/albedo data. Use a wall-loss or ablation mode.")

    bessel_data = dispatch_out[5]
    if not bessel_data:
        raise ValueError("No bessel_data available to plot albedo.")

    t_ns = np.array(sorted(bessel_data.keys()), dtype=float)
    albedo_old = np.array([bessel_data[t].get('albedo_old', np.nan) for t in t_ns], dtype=float)
    albedo_new = np.array([bessel_data[t].get('avg_albedo', np.nan) for t in t_ns], dtype=float)

    def _loess_smooth(x_vals, y_vals, span=5):
        """Simple LOESS-like local linear smoother with tricube weights."""
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        n = x_vals.size
        if n == 0:
            return y_vals

        span = int(max(3, min(span, n)))
        y_smooth = np.full(n, np.nan, dtype=float)

        for i in range(n):
            left = max(0, i - span // 2)
            right = min(n, left + span)
            left = max(0, right - span)

            xw = x_vals[left:right]
            yw = y_vals[left:right]
            valid = np.isfinite(yw)
            if np.count_nonzero(valid) < 2:
                y_smooth[i] = y_vals[i] if np.isfinite(y_vals[i]) else 0.0
                continue

            xw = xw[valid]
            yw = yw[valid]
            d = np.abs(xw - x_vals[i])
            dmax = np.max(d)
            if dmax <= 0:
                y_smooth[i] = np.mean(yw)
                continue

            u = d / dmax
            w = (1.0 - u**3) ** 3
            X = np.column_stack((np.ones_like(xw), xw))
            WX = X * w[:, None]
            beta, *_ = np.linalg.lstsq(WX, yw * w, rcond=None)
            y_smooth[i] = beta[0] + beta[1] * x_vals[i]

        return y_smooth

    span = 5
    albedo_old_smooth = _loess_smooth(t_ns, albedo_old, span=span)
    albedo_new_smooth = _loess_smooth(t_ns, albedo_new, span=span)

    plt.figure(figsize=(8, 6))
    plt.plot(t_ns, albedo_old, color='gray', linestyle='-', linewidth=1.0, alpha=0.4, label='Albedo old (raw)')
    plt.plot(t_ns, albedo_new, color='salmon', linestyle='-', linewidth=1.0, alpha=0.4, label='Albedo new (raw)')
    plt.plot(t_ns, albedo_old_smooth, color='black', linestyle='-', linewidth=2, label=f'Albedo old (LOESS span={span})')
    plt.plot(t_ns, albedo_new_smooth, color='red', linestyle='-', linewidth=2, label=f'Albedo new (LOESS span={span})')
    plt.xlabel("Time (ns)")
    plt.ylabel("Albedo")
    plt.title(f"Albedo at z=0 vs Time - Material: {Material}")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.2, max(t_ns)*1.1)
    plt.legend()
    plt.tight_layout()

    save_figure("albedo_z0_vs_time.png", model2_D=True, dpi=150, bbox_inches='tight')


def plot_model_shock_wave_at_z0_all_times(times_to_store, *, wall_material='Gold', lam_eff_power=1.5):
    """Plot model shock-front and gold-front radius at z=0 over all times and export CSV.

    Uses analytical model outputs only (no 2D numerical simulation fields).
    The shock radius is extracted from ``shock_penetration_radius_profile`` at z=0.
    The gold front radius is extracted from ``wall_penetration_radius_profile`` at z=0.
    """
    times_in = np.asarray(times_to_store, dtype=float)
    if times_in.size == 0:
        raise ValueError("times_to_store is empty.")

    front_series = compute_standard_analytic_front_series(
        times_in,
        wall_material=wall_material,
        lam_eff_power=lam_eff_power,
    )
    bessel_data_2D = front_series.get("bessel_data_2D", {})
    if not bessel_data_2D:
        raise ValueError("Model did not return bessel_data_2D; cannot extract shock profile.")

    t_ns_arr = []
    shock_radius_cm_arr = []
    gold_radius_cm_arr = []

    # Accept both second-based and ns-based time arrays.
    if float(np.nanmax(times_in)) > 1e-5:
        target_times_ns = times_in
    else:
        target_times_ns = times_in * 1e9

    for t_ns_target in target_times_ns:
        t_ns_target = float(t_ns_target)
        closest_t_ns = min(bessel_data_2D.keys(), key=lambda k: abs(k - t_ns_target))
        snapshot = bessel_data_2D[closest_t_ns]

        # Extract foam shock radius at z=0
        if "shock_penetration_radius_profile" in snapshot:
            shock_radius_cm = float(np.asarray(snapshot["shock_penetration_radius_profile"], dtype=float)[0])
        else:
            shock_radius_cm = np.nan

        # Extract gold front radius at z=0
        if "wall_penetration_radius_profile" in snapshot:
            gold_radius_cm = float(np.asarray(snapshot["wall_penetration_radius_profile"], dtype=float)[0])
        else:
            gold_radius_cm = np.nan

        t_ns_arr.append(float(closest_t_ns))
        shock_radius_cm_arr.append(shock_radius_cm)
        gold_radius_cm_arr.append(gold_radius_cm)

    t_ns_arr = np.asarray(t_ns_arr, dtype=float)
    shock_radius_cm_arr = np.asarray(shock_radius_cm_arr, dtype=float)
    gold_radius_cm_arr = np.asarray(gold_radius_cm_arr, dtype=float)
    shock_radius_mm_arr = shock_radius_cm_arr * 10.0
    gold_radius_mm_arr = gold_radius_cm_arr * 10.0

    valid = np.isfinite(t_ns_arr) & np.isfinite(shock_radius_mm_arr)
    t_ns_arr = t_ns_arr[valid]
    shock_radius_cm_arr = shock_radius_cm_arr[valid]
    shock_radius_mm_arr = shock_radius_mm_arr[valid]
    gold_radius_cm_arr = gold_radius_cm_arr[valid]
    gold_radius_mm_arr = gold_radius_mm_arr[valid]

    plt.figure(figsize=(10, 6))
    plt.plot(t_ns_arr, shock_radius_mm_arr - 0.8, color='darkred', linewidth=2.5, label='Foam shock front', marker='o', markersize=3, alpha=0.7)
    plt.plot(t_ns_arr, gold_radius_mm_arr - 0.8, color='gold', linewidth=2.5, label='Gold front (foam-gold interface)', marker='s', markersize=3, alpha=0.7)
    plt.xlabel("Time (ns)", fontsize=12)
    plt.ylabel("Radius at z=0 (mm)", fontsize=12)
    plt.title("Shock Wave and Gold Front at z=0 for All Times (Analytical Model)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    save_figure("shock_wave_at_z0_all_times_MODEL.png", model2_D=True, dpi=250, bbox_inches='tight')

    save_series_csv(
        DATA_DIR / "1.5 model" / "shock_wave_at_z0_all_times_MODEL.csv",
        {
            "time_ns": t_ns_arr,
            "shock_radius_mm": shock_radius_mm_arr,
            "shock_radius_cm": shock_radius_cm_arr,
            "gold_radius_mm": gold_radius_mm_arr,
            "gold_radius_cm": gold_radius_cm_arr,
        },
    )

    return {
        "time_ns": t_ns_arr,
        "shock_radius_mm": shock_radius_mm_arr,
        "shock_radius_cm": shock_radius_cm_arr,
        "gold_radius_mm": gold_radius_mm_arr,
        "gold_radius_cm": gold_radius_cm_arr,
    }

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

# Let's create a function that by getting a material name, it will run the appropriate comparison function for that material. This way we can easily switch between materials and their corresponding comparisons.
def compare_for_material():
    if Material == "SiO2":
        times_to_store = np.linspace(0.01, 3, 1000)
        Back_SiO2(times_to_store)
    elif Material == "C11H16Pb0.3852":
        times_to_store = np.linspace(0.01, 1, 1000)
        compare_with_article_2_exp1_Massen(times_to_store)
    elif Material == "C6H12" or Material == "C6H12Cu0.394":
        times_to_store = np.linspace(0.01, 2, 1000)
        compare_with_article_2_exp2_Xu(times_to_store)
    elif Material == "Ta2O5":
        times_to_store = np.linspace(0.01, 3, 1000)
        compare_with_article_2_exp3_13a(times_to_store)
    elif Material == "SiO2_low_energy":
        times_to_store = np.linspace(0.01, 15, 1000)
        compare_with_article_2_exp4_14(times_to_store)
    elif Material == "SiO2_Moore":
        times_to_store = np.linspace(0.01, 4, 1000)
        compare_with_article_2_exp5_15a(times_to_store)
    elif Material == "C8H7Cl":
        times_to_store = np.linspace(0.01, 4, 1000)
        compare_with_article_2_exp5_15b(times_to_store)
    elif Material == "C15H20O6" or Material == "C15H20O6Au0.172":
        times_to_store = np.linspace(0.01, 3, 1000)
        compare_with_article_2_exp6_16(times_to_store)
    elif Material == "C8H8":
        times_to_store = np.linspace(0.01, 1.5, 1000)
        compare_with_article_2_exp7_17(times_to_store)
    elif Material == "french_gold":
        times_to_store = np.linspace(0.01, 4, 1000)
        compare_with_french_gold(times_to_store)
    elif Material == "french_cupper":
        times_to_store = np.linspace(0.01, 4, 1000)
        compare_with_french_copper(times_to_store)
    else:
        print(f"No comparison function defined for material: {Material}")
        return None
    return times_to_store

if __name__ == "__main__":
    times_to_store = np.linspace(0.01, 3, 1000)
    #plot_albedo_z0_vs_time(times_to_store, mode="marshak_wall_loss", vary_rho=False, lam_eff=False, power=1.5, wall_material="Be")
    times_to_store = compare_for_material()  # times_to_store will be set inside the function based on the material
    #compare_with_marshak_results()
    #R_of_t_z(times_to_store=times_to_store)
    #compare_n_1(times_to_store)
    # plot_surface_temperature_comparison(times_to_store)
    plot_albedo_z0_vs_time(times_to_store, mode="varying_rho", vary_rho=True, lam_eff=True, power=1.5, wall_material="Gold")
    plot_model_shock_wave_at_z0_all_times(times_to_store)