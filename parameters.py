import csv
import os
import bisect
import numpy as np
import matplotlib
from csv_helpers import article_temperature_path
# Set serif fonts BEFORE importing pyplot
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Simple font configuration - just serif
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'axes.unicode_minus': False,
})

import tqdm
import pandas as pd
import scipy
# -----------------------------
# Parameters
# -----------------------------
# changing all simulation constants to cgs units.
c = 3e10        # speed of light (cm/s)
chi = 1000# global multiplier χ - big chi: LTE
a_kelvin = 7.5646e-15    # radiation constant

def load_time_temp(csv_path):
    """
    Loads time (ns) and temperature (eV) data from a CSV file.
    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    Returns
    -------
    t : np.ndarray
        1D array of times (in ns).
    T : np.ndarray
        1D array of temperatures (in eV).
    """
    t = []
    T = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # <-- skip header row
        for row in reader:
            t.append(float(row[1]))   # time
            T.append(float(row[2]))   # temperature
    return np.array(t), np.array(T)

kind_of_D_face = "arithmetic"  # "harmonic", "arithmetic", "geometric"
Material = "C8H7Cl"  # "SiO2", "Gold", "C11H16Pb0.3852", "C6H12", "C6H12Cu0.394", "Ta2O5", "Si_Moore", "C8H7Cl", "C15H20O6", "C15H20O6Au0.172", "C8H8"

if Material == "SiO2":
    Experiment = "Back"
    # self similarity model fudge factors - Foam (the first article and the first part of the second article)
    f = 8.77 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 9175      
    alpha = 3.53     # opacity exponent
    beta = 1.1       # beta exponent
    lambda_param = 0.75
    mu = 0.09
    rho = 0.05     # initial density (g/cm^3)
    R_cm = 0.08      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #T_array_TD = np.full_like(t_array_TD, 150)  # eV (the get_TD function will scale it down by 0.01)
    #lambda_ross = 0.18 # Rosseland mean free path (cm) at maximum temperature (1.8 Hev)

elif Material == "C11H16Pb0.3852":
    Experiment = "Massen"
    # self similarity model fudge factors - Pb - Massen (first experiment in VI part)
    f = 10.17 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 3200      
    alpha = 1.57       # opacity exponent
    beta = 1.2       # beta exponent
    lambda_param = 0.1
    mu = 0
    rho = 0.08      # initial density (g/cm^3)
    R_cm = 0.08      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
    t_array_TD = np.linspace(0, 1.0, 1000)  # ns
    T_array_TD = 120*np.ones_like(t_array_TD)   # eV (the get_TD function will scale it down by 0.01)
    #lambda_ross = 0.0095 # Rosseland mean free path (cm) at maximum temperature (1.5 Hev)

elif Material == "C6H12":
    Experiment = "Xu"
    # self similarity model fudge factors - CH - Xu (second experiment in VI part)
    f = 12.27 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 3926.6      
    alpha = 2.98       # opacity exponent
    beta = 1.0       # beta exponent
    lambda_param = 0.95
    mu = 0.04
    rho = 0.05      # initial density (g/cm^3)
    R_cm = 0.03      # radius of the foam cylinder (cm) - The diameter is 1.2 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.356 # Rosseland mean free path (cm) at maximum temperature (1.6 Hev)

elif Material == "C6H12Cu0.394":
    Experiment = "Xu"
    # self similarity model fudge factors - Cu - Xu (first experiment in VI part)
    f = 8.13 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 7692.9      
    alpha = 3.44       # opacity exponent
    beta = 1.1       # beta exponent
    lambda_param = 0.67
    mu = 0.07
    rho = 0.05      # initial density (g/cm^3)
    R_cm = 0.03      # radius of the foam cylinder (cm) - The diameter is 1.2 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.098 # Rosseland mean free path (cm) at maximum temperature (1.6 Hev)

elif Material == "Ta2O5":
    Experiment = "Back"
    # self similarity model fudge factors - Ta2O5 - Fig 13a
    f = 4.78 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 8433.3      
    alpha = 1.78       # opacity exponent
    beta = 1.37     # beta exponent
    lambda_param = 0.24
    mu = 0.12
    rho = 0.04      # initial density (g/cm^3)
    R_cm = 0.08     # radius of the foam cylinder (cm) - The diameter is 1.2 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.018 # Rosseland mean free path (cm) at maximum temperature (1.8 Hev)

if Material == "SiO2_Moore":
    Experiment = "Moore"
    # self similarity model fudge factors - Foam (the first article and the first part of the second article)
    f = 8.77 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 9175      
    alpha = 3.53     # opacity exponent
    beta = 1.1       # beta exponent
    lambda_param = 0.75
    mu = 0.09
    rho = 0.1249    # initial density (g/cm^3)
    R_cm = 0.1      # radius of the foam cylinder (cm) - The diameter is 2 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    T_array_TD = (340.2/368.1)**0.25 * T_array_TD  # Moore data correction
    #lambda_ross = 0.212 # Rosseland mean free path (cm) at maximum temperature (3.05 Hev)

elif Material == "C8H7Cl":
    Experiment = "Moore"
    # self similarity model fudge factors - C8H7Cl - Fig 15b
    f = 14.47 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 24466      
    alpha = 5.7       # opacity exponent
    beta = 0.96    # beta exponent
    lambda_param = 0.72
    mu = 0.04
    rho = 0.1139      # initial density (g/cm^3)
    R_cm = 0.1     # radius of the foam cylinder (cm) - The diameter is 2 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.988 # Rosseland mean free path (cm) at maximum temperature (3.05 Hev)

if Material == "SiO2_low_energy":
    Experiment = "Back"
    # self similarity model fudge factors - Foam (the first article and the first part of the second article)
    f = 8.4 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 9652      
    alpha = 2.0     # opacity exponent
    beta = 1.23       # beta exponent
    lambda_param = 0.61
    mu = 0.1
    rho = 0.01     # initial density (g/cm^3)
    R_cm = 0.15      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
    csv_path = article_temperature_path("T_bath_right.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    # T_array_TD = 100 * T_array_TD
    #lambda_ross = 0.11 # Rosseland mean free path (cm) at maximum temperature (1.8 Hev)

elif Material == "C15H20O6":
    Experiment = "Keiter"
    # self similarity model fudge factors - C15H20O6 - Fig 16 pure
    f = 11.54 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 26549      
    alpha = 5.28       # opacity exponent
    beta = 0.94  # beta exponent
    lambda_param = 0.95
    mu = 0.038
    rho = 0.065      # initial density (g/cm^3)
    R_cm = 0.04     # radius of the foam cylinder (cm) - The diameter is 0.8 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.39 # Rosseland mean free path (cm) at maximum temperature (2.1 Hev)

elif Material == "C15H20O6Au0.172":
    Experiment = "Keiter"
    # self similarity model fudge factors - C15H20O6 - Fig 16 doped with Au
    f = 9.81 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 4760      
    alpha = 2.5      # opacity exponent
    beta = 1.04    # beta exponent
    lambda_param = 0.35
    mu = 0.06
    rho = 0.0625      # initial density (g/cm^3)
    R_cm = 0.04     # radius of the foam cylinder (cm) - The diameter is 0.8 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.057 # Rosseland mean free path (cm) at maximum temperature (2.1 Hev)

elif Material == "C8H8":
    Experiment = "Ji-Yan"
    # self similarity model fudge factors - C8H8 - not used in paper
    f = 21.17 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 2818.1      
    alpha = 2.79       # opacity exponent
    beta = 1.06    # beta exponent
    lambda_param = 0.81
    mu = 0.06
    rho = 0.16     # initial density (g/cm^3)
    R_cm = 0.01     # radius of the foam cylinder (cm) - The diameter is 0.2 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.047 # Rosseland mean free path (cm) at maximum temperature (1.75 Hev)

if Material == "french_gold":
    Experiment = "French"
    # self similarity model fudge factors - Foam (the first article and the first part of the second article)
    f = 8.77 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 9175      
    alpha = 3.53     # opacity exponent
    beta = 1.1       # beta exponent
    lambda_param = 0.75
    mu = 0.09
    rho = 0.029     # initial density (g/cm^3)
    R_cm = 0.05      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.41 # Rosseland mean free path (cm) at maximum temperature (1.78 Hev)

if Material == "french_cupper":
    Experiment = "French"
    # self similarity model fudge factors - Foam (the first article and the first part of the second article)
    f = 8.77 * 10**13          # fudge factor for sigma (new model) [erg/g]
    g = 1 / 9175      
    alpha = 3.53     # opacity exponent
    beta = 1.1       # beta exponent
    lambda_param = 0.75
    mu = 0.09
    rho = 0.0189     # initial density (g/cm^3)
    R_cm = 0.1      # radius of the foam cylinder (cm) - The diameter is 1.6 mm
    csv_path = article_temperature_path("T_drive.csv")
    t_array_TD, T_array_TD = load_time_temp(csv_path)
    #lambda_ross = 0.866 # Rosseland mean free path (cm) at maximum temperature (1.78 Hev)

T_material_0_Kelvin = 300.0
eV_joule = 1.60218e-19  # J/eV
erg_per_joule = 1.0e7   # erg/J
eV = eV_joule * erg_per_joule  # erg 
Hev=1.0e2 * eV  # erg

# eV = 1.60218e-12 erg
# Hev = 1.60218e-10 erg

k_B_joule = 1.38065e-23  # J/K
k_B = k_B_joule * erg_per_joule  # erg/K
K_per_Hev = Hev / k_B  # in HeV (kelvin per HeV)

# k_B = 1.3805e-16 erg/K
# K_per_Hev = 1.1605e6 K/HeV

a_hev = a_kelvin * (K_per_Hev**4)  # radiation constant in HeV units, equles 7.5646e-15 * (1.1605e6)^4 = 1.374e10 HeV^3/cm^3
T_material_0_hev = T_material_0_Kelvin / K_per_Hev

# -----------------------------
# Grid and time step
# -----------------------------
if Material == "SiO2":
    L = 0.3      
    Nz = 500   # increase resolution because domain is much larger
elif Material == "SiO2_low_energy":
    L = 0.2      
    Nz = 500 
elif Material == "Gold":
    L = 0.0003    # cm
    Nz = 1000
elif Material == "C11H16Pb0.3852":
    L = 0.03    # cm
    Nz = 500
elif Material == "C6H12" or Material == "C6H12Cu0.394":
    L = 0.05    # cm
    Nz = 500
elif Material == "Ta2O5":
    L = 0.2      
    Nz = 500   # increase resolution because domain is much larger
elif Material == "C8H7Cl" or Material == "SiO2_Moore":
    L = 0.28      
    Nz = 500   # increase resolution because domain is much larger
elif Material == "C15H20O6" or Material == "C15H20O6Au0.172":
    L = 0.12      
    Nz = 500   # increase resolution because domain is much larger
elif Material == "C8H8":
    L = 0.03      
    Nz = 500   # increase resolution because domain is much larger
elif Material == "french_gold":
    L = 0.2      
    Nz = 500   # increase resolution because domain is much larger
elif Material == "french_cupper":
    L = 0.4      
    Nz = 500   # increase resolution because domain is much larger

# self similarity model fudge factors - gold
f_gold = 3.4 * 10**13          # fudge factor for sigma (new model) [erg/g]
g_gold = 1 / 7200      
alpha_gold = 1.5       # opacity exponent
beta_gold = 1.6       # beta exponent
lambda_param_gold = 0.2
mu_gold = 0.14
rho_gold = 19.32      # initial density (g/cm^3)

# self similarity model fudge factors - Be     
alpha_be = 4.893       # opacity exponent
beta_be = 1.0902      # beta exponent
lambda_param_be = 0.6726
mu_be = 0.0701
rho_be = 1.85      # initial density (g/cm^3)

# self similarity model fudge factors - Copper 
alpha_copper = 2.21       # opacity exponent
beta_copper = 1.35     # beta exponent
lambda_param_copper = 0.29
mu_copper = 0.14
rho_copper = 8.96      # initial density (g/cm^3)

z = np.linspace(0.0, L, Nz)
dz = z[1] - z[0]

# Radial grid for 2D cylindrical (Foam)
Nr = 1000
r_grid = np.linspace(0.0, R_cm, Nr)
dr = r_grid[1] - r_grid[0]


def solve_q_from_dr0(gold_width, N, dr0):
    """
    Solve q >= 1 such that sum_{k=0}^{N-1} dr0*q^k = gold_width.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    if dr0 <= 0:
        raise ValueError("dr0 must be > 0")
    if gold_width <= 0:
        raise ValueError("gold_width must be > 0")
    if dr0 * N > gold_width:
        print(f"dr0*N = {dr0*N} exceeds gold_width = {gold_width}")
        raise ValueError("dr0 too large: even uniform widths N*dr0 exceed gold_width")

    # Uniform special case.
    if abs(dr0 * N - gold_width) / gold_width < 1e-12:
        return 1.0

    def S(q):
        return dr0 * (q**N - 1.0) / (q - 1.0)

    q_lo = 1.0 + 1e-12
    q_hi = 2.0
    while S(q_hi) < gold_width:
        q_hi *= 2.0
        if q_hi > 1e6:
            raise RuntimeError("Failed to bracket q; check inputs.")

    for _ in range(80):
        q_mid = 0.5 * (q_lo + q_hi)
        if S(q_mid) < gold_width:
            q_lo = q_mid
        else:
            q_hi = q_mid

    return 0.5 * (q_lo + q_hi)


def make_r_two_block(R_foam, gold_width, Nr_foam, Nr_gold, dr0=None):
    """
    Build radial nodes for:
      - foam region [0, R_foam]: uniform with Nr_foam nodes
      - gold region [R_foam, R_foam+gold_width]: geometric widths

    dr0 is the first gold cell width at the foam-gold interface.
    """
    if Nr_foam < 2:
        raise ValueError("Nr_foam must be >= 2")
    if Nr_gold < 1:
        raise ValueError("Nr_gold must be >= 1")
    if R_foam <= 0 or gold_width <= 0:
        raise ValueError("R_foam and gold_width must be > 0")
    if dr0 is None:
        raise ValueError("Provide dr0=...")

    r_foam = np.linspace(0.0, R_foam, Nr_foam)

    q = solve_q_from_dr0(gold_width, Nr_gold, dr0)
    widths = dr0 * (q ** np.arange(Nr_gold))
    r_gold_block = R_foam + np.concatenate(([0.0], np.cumsum(widths)))

    R_total = R_foam + gold_width
    r_gold_block[-1] = R_total

    r = np.concatenate((r_foam, r_gold_block))
    r = np.unique(r)

    info = {"q": float(q), "dr0": float(dr0), "widths": widths, "R_total": R_total}
    return r, info

# Radial grid for 2D cylindrical (Gold extension)
Nr_gold = 100
w_Au = 25e-4
dr0_gold = w_Au / 4000
r_gold, r_gold_info = make_r_two_block(R_cm, w_Au, Nr, Nr_gold, dr0=dr0_gold)


t_final_sec = 3.73e-9 
dt_sec = 5e-15
t_final_ns = t_final_sec * 10**9
dt_ns = dt_sec * 10**9

t_final = t_final_sec
dt = dt_sec 

Nt = int(t_final / dt) + 1

# --------------------------------------------------
# Nearest-neighbor lookup in time
# --------------------------------------------------
def get_TD(t_query, t, T):
    """
    Returns the drive temperature of the whose time is closest 
    to t_query (T_D is the drive temperature extracted from the image).
    use: get_TD(t_query, t, T)
    Parameters
    ----------
    t_query : float
        The query time (in ns).
    t : np.ndarray
        1D array of times (in ns).
    T : np.ndarray
        1D array of temperatures (in eV).
    Returns
    -------
    float
        The temperature (in eV) whose time is closest to t_query.
    """
    #The index i where t_query should be inserted so that t remains sorted, 
    # inserting it on the LEFT of any equal values.
    i = bisect.bisect_left(t, t_query) 

    if i == 0:
        return 0.01 * T[0]
    if i == len(t):
        return 0.01 * T[-1]
    returned_T = T[i-1] if abs(t_query - t[i-1]) <= abs(t_query - t[i]) else T[i]
    return 0.01 * returned_T  # convert eV to HeV

T_bath_hev = get_TD(0, t_array_TD, T_array_TD)
T_bath_kelvin = T_bath_hev * K_per_Hev
# -----------------------------
# Simulation unit system
# -----------------------------
CGS = "cgs"
HEV_NS = "hev|ns"
simulation_unit_system = CGS  # CGS or HEV_NS
if simulation_unit_system == CGS:
    T_material_0 = T_material_0_Kelvin
    T_bath = T_bath_kelvin
    a = a_kelvin
    dt = dt_sec
    t_final = t_final_sec
    c = 3e10  # speed of light in cm/s
elif simulation_unit_system == HEV_NS:
    T_material_0 = T_material_0_hev
    T_bath = T_bath_hev
    a = a_hev
    dt = dt_ns
    t_final = t_final_ns
    c = 30  # speed of light in cm/ns

####################################
# relative timestep
####################################
def update_dt_relchange(dt, E, Eold, UR, URold, *, dtfac=0.05, dtmax=5e-13, growth_cap=1.1):
    """
    Adaptive dt based on max relative change in E and UR.

    dtfac: target relative change per step (~0.05 means ~5%)
    dtmax: absolute cap on dt
    growth_cap: allow dt to increase by at most 10% per step (1.1)
    """
    # Protect from division by tiny numbers
    E_min = np.max(np.abs(E)) * 1e-3 + 1e-30
    dE = np.max(np.abs(E - Eold) / (np.abs(E) + E_min))

    U_min = np.max(np.abs(UR)) * 1e-3 + 1e-30
    dU = np.max(np.abs(UR - URold) / (np.abs(UR) + U_min))

    # Avoid blow-ups if change is ~0
    dE = max(dE, 1e-16)
    dU = max(dU, 1e-16)

    dttag1 = dt / dE * dtfac
    dttag2 = dt / dU * dtfac

    dt_new = min(dttag1, dttag2, growth_cap * dt)
    if dtmax is not None:
        dt_new = min(dt_new, dtmax)

    return dt_new, dE, dU


# -----------------------------
# Simulation helpers
# -----------------------------
def init_state():
    """Initialize E, UR based on T_material_0."""
    E0 = a * T_material_0**4 * np.ones(Nz)
    UR0 = a * T_material_0**4 * np.ones(Nz)
    return E0, UR0