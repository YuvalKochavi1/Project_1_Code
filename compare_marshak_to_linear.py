from parameters import *
from model_main import *
from simulation import *
from comparison import run_case

from csv_helpers import *
DATA_DIR = BASE_DIR / "data"
FIGURES_OUTPUT_DIR = BASE_DIR / "figures"

def data_for_comparison():
    x_vals = np.array([0, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 15, 20])
    E_data = {
        0.01: np.array([
            0.09040, 0.03241, 0.00361, 0.00001,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        ]),
        0.1: np.array([
            0.24023, 0.18003, 0.11024, 0.04111, 0.01217, 0.00280,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        ]),
        1.0: np.array([
            0.46599, 0.42133, 0.36020, 0.27323, 0.20332, 0.14837,
            0.01441, 0.00005, 0.00001, np.nan, np.nan, np.nan
        ]),
        10.0: np.array([
            0.73611, 0.71338, 0.67978, 0.62523, 0.57274, 0.52255,
            0.27705, 0.07075, 0.01271, 0.00167, 0.00002, np.nan
        ])
}
    U_data = {
        0.01: np.array([
        0.00062, 0.00014, 0.00001,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ]),
    0.1: np.array([
        0.01641, 0.01068, 0.00532, 0.00143, 0.00032, 0.00005,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ]),
    1.0: np.array([
        0.24762, 0.21614, 0.17530, 0.12182, 0.08306, 0.05556,
        0.00324, 0.00001, np.nan, np.nan, np.nan, np.nan
    ]),
    10.0: np.array([
        0.72328, 0.69946, 0.66432, 0.60749, 0.55308, 0.50134,
        0.25413, 0.05936, 0.00968, 0.00115, 0.00001, np.nan
        ]),
    }
    return x_vals, E_data, U_data

def compare_with_marshak_results():
    # changing global parameters for comparison
    global alpha, lambda_param, g, f, mu, beta, L, dt, t_final, Nz, z, dz, T_bath_kelvin, T_bath, chi,\
            T_material_0, eps, sigma, show_plots, Nt, z, dz
    show_plots = False
    x_vals, E_data, U_data = data_for_comparison()

    # setting self-similar model parameters for comparison with linear theory
    alpha = 0
    lambda_param = -1
    g = 1
    f = a_hev
    mu = 1
    beta = 4

    # Boundary and initial conditions
    c = 3.0e10        # speed (cm/s or arbitrary)
    sigma = 1      # opacity Σ (1/cm)
    eps = 1        # ε in your equation
    T_material_0 = 300      # initial matter "temperature" or energy U
    T_bath = 1000000     # You must change the T_bath is the simulation to match this
    L = 10          # length of the slab (cm)
    Nz = 5000     # number of spatial points
    t_final = (1.0e-9)/3  # final time (s)
    a = 7.5646e-15  # radiation constant
    
    z = np.linspace(0, L, Nz)
    dz = z[1] - z[0]
    t_final_sec = (1.0e-9)/3  # final time (s)
    dt_sec = 1.0e-12 # initial guess for time step (s)
    Nt = int(t_final / dt) + 1  # number of time steps (dimensional)
    chi = 1.0  # coupling coefficient
    t_final = t_final_sec
    dt = dt_sec 

    times_to_store = np.array([0.01, 0.1, 1.0, 10.0]) / c  # in s
    _  = run_case(times_to_store=times_to_store, reset_initial_conditions=True, marshak_boundary=True)
    stored_U = pd.read_csv(DATA_DIR / "stored_Um_marshak.csv", header=None).to_numpy()
    stored_t = pd.read_csv(DATA_DIR / "stored_time_marshak.csv", header=None).to_numpy().flatten()
    stored_TR = pd.read_csv(DATA_DIR / "stored_TR_marshak.csv", header=None).to_numpy()
    stored_TR_KELVIN = stored_TR * K_per_Hev
    stored_E = stored_TR_KELVIN ** 4 * a_kelvin
    colors = plt.cm.viridis(np.linspace(0, 1, len(stored_t)))
    plt.figure(figsize=(10, 6))
    for E, Ui, ti, color in zip(stored_E, stored_U, times_to_store, colors):
        plt.scatter(x_vals/np.sqrt(3), U_data[ti*c], label=f"Marshak Theory t={ti:.1e} s", marker='x', color=color)
        plt.scatter(x_vals/np.sqrt(3), E_data[ti*c], label=f"Marshak Theory t={ti:.1e} s", marker='o', color=color)
        plt.plot(z, Ui / (a * T_bath**4), label=f"t={ti:.1e} s", color=color, linestyle='--')
        plt.plot(z, E / (a*T_bath**4), label=f"t={ti:.1e} s", color=color, linestyle='-')

    plt.xlabel("z (cm)")
    plt.ylabel(r"$T(z,t)/T_{\mathrm{bath}}$")
    plt.xscale("log")
    plt.xlim(1e-2, L)
    plt.grid(True)
    plt.legend()
    plt.title(f"Temperature profiles over time (Material={Material})")
    plt.tight_layout()
    # make directory if not exists
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
    plt.savefig(FIGURES_OUTPUT_DIR / "comparison_marshak.png")