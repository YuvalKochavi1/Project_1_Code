"""
2D (z,r) diffusion solver (implicit, backward Euler) - only for the Foam material model for now, but can be adapted to other Material blocks 
by changing the material model hooks (sigma_of_T, beta_of_T, D_of_T, U_m_of_UR), there is no coating around the foam in this simulation.
-----------------------------------------------------------------
This is the 2D generalization of your 1D self-similar implicit scheme:

- Variable diffusion coefficient: D(T) = c/(3*sigma(T))
- Face diffusion uses harmonic/arithmetic/geometric average
- Implicit coupling via A = beta(T)*dt*chi*c*sigma(T)
  UR^{n+1} = (A*E^{n+1} + UR^n)/(1 + A)

Geometry:
- z in [0, Lz], r in [0, R]
- cylindrical axis at r=0 uses symmetry: dE/dr = 0  (implemented by mirroring j=-1 -> j=1)
- You can choose Dirichlet or Neumann at r=R and at z=Lz.

Implementation:
- Builds one sparse linear system for E^{n+1} each step: (I/dt - div(D grad) + coupling_diag) E^{n+1} = rhs
- Uses scipy.sparse.linalg.spsolve (direct). For large grids, switch to CG + preconditioner.

Keep your material model hooks sigma_of_T(T), beta_of_T(T), etc.
"""

import csv
import bisect
import numba as nb
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, gmres, bicgstab, spilu, LinearOperator

# -----------------------------
# Constants & unit conversions
# -----------------------------
c_cgs = 3e10  # cm/s
a_kelvin = 7.5646e-15  # erg/cm^3/K^4

eV_joule = 1.60218e-19
erg_per_joule = 1.0e7
eV = eV_joule * erg_per_joule
Hev = 1.0e2 * eV

k_B_joule = 1.38065e-23
k_B = k_B_joule * erg_per_joule
K_per_Hev = Hev / k_B

a_hev = a_kelvin * (K_per_Hev ** 4)

# -----------------------------
# Load drive temperature
# -----------------------------
def load_time_temp(csv_path):
    t, T = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            t.append(float(row[1]))  # ns
            T.append(float(row[2]))  # eV
    return np.array(t), np.array(T)

def get_TD(t_query_ns, t_ns, T_eV):
    i = bisect.bisect_left(t_ns, t_query_ns)
    if i == 0:
        return 0.01 * T_eV[0]
    if i == len(t_ns):
        return 0.01 * T_eV[-1]
    Tret = T_eV[i-1] if abs(t_query_ns - t_ns[i-1]) <= abs(t_query_ns - t_ns[i]) else T_eV[i]
    return 0.01 * Tret  # eV -> HeV

# -----------------------------
# Adaptive dt (same idea as yours)
# -----------------------------
def update_dt_relchange(dt, E, Eold, UR, URold, *, dtfac=0.05, dtmax=None, growth_cap=1.1):
    E_min = np.max(np.abs(E)) * 1e-3 + 1e-30
    dE = np.max(np.abs(E - Eold) / (np.abs(E) + E_min))
    U_min = np.max(np.abs(UR)) * 1e-3 + 1e-30
    dU = np.max(np.abs(UR - URold) / (np.abs(UR) + U_min))

    dE = max(dE, 1e-16)
    dU = max(dU, 1e-16)

    dttag1 = dt / dE * dtfac
    dttag2 = dt / dU * dtfac

    dt_new = min(dttag1, dttag2, growth_cap * dt)
    if dtmax is not None:
        dt_new = min(dt_new, dtmax)
    return dt_new, dE, dU

# ============================================================
# 2D Solver
# ============================================================
class SelfSimilarDiffusion2D:
    def __init__(
        self,
        *,
        # grid
        Lz, R, Nz, Nr,
        # time
        dt_init, t_final,
        # unit system
        simulation_unit_system="cgs",  # "cgs" or "hev|ns"
        # material params (your self-similar)
        f, g, alpha, beta_exp, lambda_param, mu, rho,
        chi=1000.0,
        # drive
        t_drive_ns=None, T_drive_eV=None,
        # face averaging
        kind_of_D_face="arithmetic",  # harmonic/arithmetic/geometric
        # initial material temperature
        T_material_0_K=300.0,
    ):
        self.Lz, self.R = float(Lz), float(R)
        self.Nz, self.Nr = int(Nz), int(Nr)
        self.z = np.linspace(0.0, self.Lz, self.Nz)
        self.r = np.linspace(0.0, self.R, self.Nr)
        self.dz = self.z[1] - self.z[0]
        self.dr = self.r[1] - self.r[0]

        self.simulation_unit_system = simulation_unit_system
        self.kind_of_D_face = kind_of_D_face

        self.f = float(f)
        self.g = float(g)
        self.alpha = float(alpha)
        self.beta_exp = float(beta_exp)
        self.lambda_param = float(lambda_param)
        self.mu = float(mu)
        self.rho = float(rho)
        self.chi = float(chi)

        self.dt_init = float(dt_init)
        self.t_final = float(t_final)

        self.t_drive_ns = t_drive_ns
        self.T_drive_eV = T_drive_eV

        # set unit-dependent constants
        if simulation_unit_system == "cgs":
            self.a = a_kelvin
            self.c = c_cgs
            self.T0 = float(T_material_0_K)  # Kelvin
        elif simulation_unit_system == "hev|ns":
            self.a = a_hev
            self.c = 30.0  # cm/ns
            self.T0 = float(T_material_0_K) / K_per_Hev  # HeV
        else:
            raise ValueError("simulation_unit_system must be 'cgs' or 'hev|ns'.")

        # allocate state
        self.E = self.a * (self.T0 ** 4) * np.ones((self.Nz, self.Nr))
        self.UR = self.a * (self.T0 ** 4) * np.ones((self.Nz, self.Nr))

    # -----------------------------
    # Material model hooks
    # -----------------------------
    def sigma_of_T(self, T):
        """
        sigma(T):
        1/sigma = g * T^alpha * rho^(-lambda-1)
        Note: in your CGS version you convert Kelvin -> HeV for sigma power-law.
        """
        if self.simulation_unit_system == "cgs":
            T_Hev = T / K_per_Hev
            return 1.0 / (self.g * (T_Hev ** self.alpha) * (self.rho ** (-self.lambda_param - 1)))
        else:
            return 1.0 / (self.g * (T ** self.alpha) * (self.rho ** (-self.lambda_param - 1)))

    def beta_of_T(self, T):
        """
        beta(T) = Cv_R / Cv_m with your conventions.
        Uses your same CGS/HeV logic.
        """
        if self.simulation_unit_system == "cgs":
            Cv_m = self.f * self.beta_exp * (T ** (self.beta_exp - 1)) * (self.rho ** (-self.mu + 1))
            Cv_R = 4.0 * self.a * (T ** 3)
            return (Cv_R / Cv_m) * (K_per_Hev ** self.beta_exp)
        else:
            return ((4.0 * self.a * (self.rho ** (self.mu - 1))) / (self.f * self.beta_exp)) * (T ** (4.0 - self.beta_exp))

    def D_of_T(self, T):
        return self.c / (3.0 * self.sigma_of_T(T))

    def U_m_of_UR(self, UR):
        # Used only for diagnostics/energy integrals
        T = (UR / self.a) ** 0.25
        if self.simulation_unit_system == "cgs":
            T_Hev = T / K_per_Hev
        else:
            T_Hev = T
        return self.f * (T_Hev ** self.beta_exp) * (self.rho ** (-self.mu + 1))

    # -----------------------------
    # Boundary conditions for E
    # -----------------------------
    def E_left_drive(self, t):
        # z=0 boundary (drive)
        if self.t_drive_ns is None or self.T_drive_eV is None:
            # fallback: constant bath
            T = self.T0
        else:
            if self.simulation_unit_system == "cgs":
                t_ns = t * 1e9
                T_hev = get_TD(t_ns, self.t_drive_ns, self.T_drive_eV)  # HeV
                T = T_hev * K_per_Hev  # Kelvin
            else:
                # in hev|ns, t is ns already
                T = get_TD(t, self.t_drive_ns, self.T_drive_eV)  # HeV
        return self.a * (T ** 4)

    def E_right_bath(self):
        # z=Lz boundary (simple)
        if self.simulation_unit_system == "cgs":
            T = 300.0
        else:
            T = 300.0 / K_per_Hev
        return self.a * (T ** 4)

    # ============================================================
    # Implicit step: build and solve sparse system for E^{n+1}
    # ============================================================
    def implicit_step(self, *, t, dt_local, bc_r_outer="dirichlet_bath", marshak_boundary=False):
        """
        bc_r_outer: "neumann0" (dE/dr=0 at r=R) or "dirichlet_bath"
        Axis r=0 always uses neumann symmetry.
        z=0: Dirichlet drive, z=Lz: Dirichlet bath
        """
        Nz, Nr = self.Nz, self.Nr
        dz, dr = self.dz, self.dr

        E_old = self.E
        UR_old = self.UR

        # lagged coefficients from UR^n
        Tn = (UR_old / self.a) ** 0.25
        Dn = self.D_of_T(Tn)
        betan = self.beta_of_T(Tn)
        sigman = self.sigma_of_T(Tn)

        # coupling diag
        A = betan * dt_local * self.chi * self.c * sigman
        coupling = self.chi * self.c * sigman / (1.0 + A)  # χ c σ/(1+A)


        # Face diffusion (z-faces and r-faces)
        # z-face: between i and i+1, shape (Nz-1, Nr)
        if self.kind_of_D_face == "harmonic":
            Dz_face = 2.0 * Dn[:-1, :] * Dn[1:, :] / (Dn[:-1, :] + Dn[1:, :] + 1e-30)
            Dr_face = 2.0 * Dn[:, :-1] * Dn[:, 1:] / (Dn[:, :-1] + Dn[:, 1:] + 1e-30)
        elif self.kind_of_D_face == "arithmetic":
            Dz_face = 0.5 * (Dn[:-1, :] + Dn[1:, :])
            Dr_face = 0.5 * (Dn[:, :-1] + Dn[:, 1:])
        elif self.kind_of_D_face == "geometric":
            Dz_face = np.sqrt(Dn[:-1, :] * Dn[1:, :])
            Dr_face = np.sqrt(Dn[:, :-1] * Dn[:, 1:])
        else:
            raise ValueError("kind_of_D_face must be harmonic/arithmetic/geometric")

        # Build sparse matrix for interior unknowns (all nodes except z boundaries if Dirichlet)
        if marshak_boundary:
            # We'll solve for all (i,j) with i=0..Nz-2, j=0..Nr-1  (include axis and outer r)
            i0, i1 = 0, Nz - 2 # z index range for interior unknowns (not included boundaries)
        else:
            # We'll solve for all (i,j) with i=1..Nz-2, j=0..Nr-1  (exclude both z boundaries)
            i0, i1 = 1, Nz - 2 # z index range for interior unknowns (not included boundaries)
        nzi = (i1 - i0 + 1)
        n_unknown = nzi * Nr

        def idx(i, j):
            # i in [i0..i1], j in [0..Nr-1]
            return (i - i0) * Nr + j

        M = lil_matrix((n_unknown, n_unknown), dtype=np.float64) # sparse matrix for E^{n+1} unknowns
        b = np.zeros(n_unknown, dtype=np.float64) # right-hand side

        E_left = self.E_left_drive(t)
        E_right = self.E_right_bath()

        r = self.r  # r_j

        for i in range(i0, i1 + 1):
            for j in range(0, Nr):
                row = idx(i, j)

                if i == 0 and marshak_boundary:
                    # Marshak boundary at z=0
                    D_ph = Dz_face[0, j]      # D_{1/2}
                    alpha = 2.0 * D_ph / (self.c * dz)
                    row = idx(0, j)
                    # (1 + alpha) E0 - alpha E1 = E_D
                    M[row, row] += (1.0 + alpha)
                    col = idx(1, j)
                    M[row, col] += -alpha
                    b[row] = self.E_left_drive(t + dt_local)
                    continue

                # diagonal starts with time + coupling
                diag = (1.0 / dt_local) + coupling[i, j]
                rhs = (E_old[i, j] / dt_local) + coupling[i, j] * UR_old[i, j]

                # -----------------------
                # z diffusion (standard 2nd derivative using face D)
                # flux coeffs: D_{i+1/2}/dz^2, D_{i-1/2}/dz^2
                # -----------------------
                D_iph = Dz_face[i, j]     # between i and i+1  (since Dz_face index is i for i+1/2)
                D_imh = Dz_face[i - 1, j] # between i-1 and i

                # neighbor i+1
                if i + 1 <= Nz - 2:
                    col = idx(i + 1, j)
                    M[row, col] += -D_iph / dz**2
                else:
                    # i+1 == Nz-1 is Dirichlet boundary at z=Lz
                    rhs += (D_iph / dz**2) * E_right
                diag += (D_iph + D_imh) / dz**2

                # neighbor i-1
                if i - 1 >= i0:
                    col = idx(i - 1, j)
                    M[row, col] += -D_imh / dz**2
                else:
                    if not marshak_boundary:
                        # i-1 == 0 is Dirichlet boundary at z=0
                        rhs += (D_imh / dz**2) * E_left       
                # -----------------------
                # r diffusion in cylindrical form:
                # (1/r) d/dr ( r D dE/dr )
                # Discretize with conservative fluxes at r_{j±1/2}:
                # term ≈ [ r_{j+1/2} D_{j+1/2} (E_{j+1}-E_j)/dr
                #        - r_{j-1/2} D_{j-1/2} (E_j-E_{j-1})/dr ] / (r_j dr)
                #
                # Handle r=0 by symmetry: E_{j-1} = E_{1} when j=0 (Neumann).
                # -----------------------
                if Nr > 1:
                    if j == 0:
                        # axis: impose dE/dr=0 -> E_{-1} = E_{1}
                        # Equivalent conservative form with r_{-1/2}=0:
                        # flux at r_{-1/2} is zero, only + side contributes
                        rj = r[j] + 1e-30  # avoid divide by 0; but formula below uses r_{1/2}/(r0*dr^2)
                        r_ph = 0.5 * (r[0] + r[1])  # r_{1/2} = dr/2
                        D_ph = Dr_face[i, 0]        # between j=0 and j=1
                        coeff = (r_ph * D_ph) / (rj * dr**2)

                        # neighbor j+1
                        col = idx(i, 1)
                        M[row, col] += -coeff # a returning boundary condition for j=0
                        diag += coeff
                        # no j-1 term at axis (flux=0)

                    elif j == Nr - 1:
                        # outer boundary at r=R
                        if bc_r_outer == "neumann0":
                            # dE/dr=0 -> E_{Nr} = E_{Nr-2}
                            rj = r[j]
                            r_mh = 0.5 * (r[j-1] + r[j])
                            D_mh = Dr_face[i, j-1]
                            coeff_mh = (r_mh * D_mh) / (rj * dr**2)

                            # j-1 neighbor
                            col = idx(i, j - 1)
                            M[row, col] += -coeff_mh # a returning boundary condition for j=Nr-1
                            diag += coeff_mh
                            # + side flux is zero

                        elif bc_r_outer == "dirichlet_bath":
                            # E(r=R) fixed to bath value
                            E_rR = E_right
                            rj = r[j]
                            r_mh = 0.5 * (r[j-1] + r[j])
                            D_mh = Dr_face[i, j-1]
                            coeff_mh = (r_mh * D_mh) / (rj * dr**2)

                            # j-1 neighbor
                            col = idx(i, j - 1)
                            M[row, col] += -coeff_mh
                            diag += coeff_mh

                            # + side: r_{j+1/2} ~ r_j + dr/2, D_{+} ~ D_mh (simple)
                            r_ph = r[j] + 0.5 * dr
                            D_ph = D_mh
                            coeff_ph = (r_ph * D_ph) / (rj * dr**2)
                            rhs += coeff_ph * E_rR
                            diag += coeff_ph
                        else:
                            raise ValueError("bc_r_outer must be 'neumann0' or 'dirichlet_bath'.")

                    else:
                        rj = r[j]
                        r_ph = 0.5 * (r[j] + r[j + 1])
                        r_mh = 0.5 * (r[j - 1] + r[j])
                        D_ph = Dr_face[i, j]     # between j and j+1
                        D_mh = Dr_face[i, j - 1] # between j-1 and j

                        coeff_ph = (r_ph * D_ph) / (rj * dr**2)
                        coeff_mh = (r_mh * D_mh) / (rj * dr**2)

                        # j+1
                        col = idx(i, j + 1)
                        M[row, col] += -coeff_ph
                        # j-1
                        col = idx(i, j - 1)
                        M[row, col] += -coeff_mh

                        diag += (coeff_ph + coeff_mh)

                if not(i == 0 and marshak_boundary): # for Marshak, the i=0 row is written separately above with the BC, so skip writing it here to avoid overwrite
                    # write diagonal and rhs
                    M[row, row] += diag
                    b[row] = rhs

        # solve for interior E
        E_inner = spsolve(csr_matrix(M), b)

        # reconstruct full E^{n+1} with Dirichlet in z
        E_new = E_old.copy()
        if not marshak_boundary:
            E_new[0, :] = E_left
        E_new[-1, :] = E_right
        for i in range(i0, i1 + 1):
            for j in range(Nr):
                E_new[i, j] = E_inner[idx(i, j)]

        # enforce r-axis symmetry explicitly (helps numeric noise)
        if Nr > 1:
            E_new[:, 0] = E_new[:, 1]

        # update UR implicitly (local)
        UR_new = (A * E_new + UR_old) / (1.0 + A)

        self.E = E_new
        self.UR = UR_new

    # ============================================================
    # Time loop with storage
    # ============================================================
    def run(self, times_to_store, *, dtfac=0.05, dtmin=None, dtmax=None, bc_r_outer="dirichlet_bath", marshak_boundary=False):
        times_to_store = np.array(times_to_store, dtype=float)

        stored_t = []
        stored_Um = []
        stored_Tm = []
        stored_TR = []

        t = 0.0
        dt_local = self.dt_init
        store_idx = 0

        pbar = tqdm.tqdm(total=self.t_final, desc="Simulating 2D", unit="s", ncols=100)

        while t < self.t_final - 1e-30:
            dt_local = min(dt_local, self.t_final - t)

            if store_idx < len(times_to_store):
                t_target = times_to_store[store_idx]
                if t < t_target <= t + dt_local:
                    dt_local = t_target - t

            Eold = self.E.copy()
            URold = self.UR.copy()

            self.implicit_step(t=t, dt_local=dt_local, bc_r_outer=bc_r_outer, marshak_boundary=marshak_boundary)
            t_next = t + dt_local

            Um = self.U_m_of_UR(self.UR)
            Tm = (self.UR / self.a) ** 0.25
            TR = (self.E / self.a) ** 0.25
            if store_idx < len(times_to_store) and abs(t_next - times_to_store[store_idx]) < 0.5*dt_local:
                stored_t.append(t_next)
                stored_Um.append(Um.copy())
                stored_Tm.append(Tm.copy())
                stored_TR.append(TR.copy())
                store_idx += 1

            dt_new, dE, dU = update_dt_relchange(dt_local, self.E, Eold, self.UR, URold, dtfac=dtfac, dtmax=dtmax)
            if dtmin is not None:
                dt_new = max(dt_new, dtmin)

            pbar.update(t_next - t)
            t = t_next
            dt_local = dt_new

        pbar.close()
        return np.array(stored_t), np.array(stored_Um), np.array(stored_Tm), np.array(stored_TR)

# ============================================================
# Example usage (adapt to your Material blocks)
# ============================================================
if __name__ == "__main__":
    Material = "SiO2"

    if Material == "Gold":
        f = 3.4e13
        g = 1 / 7200
        alpha = 1.5
        beta_exp = 1.6
        lambda_param = 0.2
        mu = 0.14
        rho = 19.32

        # Drive CSV (optional)
        csv_path = "Project 1/data_articles_1_2/article_1_tempertures/T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)
        print("Loaded drive data points:", len(t_drive_ns))

        # Geometry / grid
        Lz = 0.0003  # cm
        R = 0.00015  # cm  (choose)
        Nz = 20    # start smaller; grow later
        Nr = 20

        # Time
        t_final = 3e-9
        dt_init = 5e-15

        sim = SelfSimilarDiffusion2D(
            Lz=Lz, R=R, Nz=Nz, Nr=Nr,
            dt_init=dt_init, t_final=t_final,
            simulation_unit_system="cgs",
            f=f, g=g, alpha=alpha, beta_exp=beta_exp, lambda_param=lambda_param, mu=mu, rho=rho,
            chi=1000.0,
            t_drive_ns=t_drive_ns, T_drive_eV=T_drive_eV,
            kind_of_D_face="arithmetic",  # harmonic/arithmetic/geometric
            T_material_0_K=300.0,
        )

        times_to_store = np.linspace(0.01, t_final, 50)
        stored_t, stored_Um, stored_Tm, stored_TR = sim.run(
            times_to_store,
            dtfac=0.05,
            dtmin=5e-15,
            dtmax=2e-13,
            bc_r_outer="dirichlet_bath",  # or "neumann0"
        )

        # Example diagnostic: front location along z at r=0
        z = sim.z
        fronts = []
        for Tm in stored_Tm:
            # crude “front”: max gradient in z at r=0
            prof = Tm[:, 0]
            front_idx = np.argmax(np.abs(np.diff(prof)))
            fronts.append(z[front_idx])
        fronts_arr = np.array(fronts, dtype=float)
    
    if Material == "SiO2":
        # self similarity model fudge factors - Foam (the first article and the first part of the second article)
        f = 8.77 * 10**13          # fudge factor for sigma (new model) [erg/g]
        g = 1 / 9175      
        alpha = 3.53     # opacity exponent
        beta_exp = 1.1       # beta exponent
        lambda_param = 0.75
        mu = 0.09
        rho = 0.05     # initial density (g/cm^3)
        csv_path = "Project 1/data_articles_1_2/article_1_tempertures/T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)

        # Geometry / grid
        Lz = 0.3  # cm
        R = 0.08  # cm  (choose)
        Nz = 20  # start smaller; grow later
        Nr = 20

        # Time
        t_final = 3e-9
        dt_init = 5e-15

        sim = SelfSimilarDiffusion2D(
            Lz=Lz, R=R, Nz=Nz, Nr=Nr,
            dt_init=dt_init, t_final=t_final,
            simulation_unit_system="cgs",
            f=f, g=g, alpha=alpha, beta_exp=beta_exp, lambda_param=lambda_param, mu=mu, rho=rho,
            chi=1000.0,
            t_drive_ns=t_drive_ns, T_drive_eV=T_drive_eV,
            kind_of_D_face="arithmetic",  # harmonic/arithmetic/geometric
            T_material_0_K=300.0,
        )

        times_to_store = t_final * np.linspace(0.01, 1.0, 50)
        stored_t, stored_Um, stored_Tm, stored_TR = sim.run(times_to_store,dtfac=0.05,dtmin=5e-15,dtmax=2e-13,bc_r_outer="dirichlet_bath",marshak_boundary=True) # "neumann0" or "dirichlet_bath" for r outer BC

        # Example diagnostic: front location along r at z=0
        z = sim.z
        fronts = []
        for Tm in stored_Tm:
            # crude “front”: max gradient in z at r=0
            prof = Tm[:, 0]
            min_val = prof[0]
            #add to fronts the smallest temp bigger than 300K
            front_idx = np.argmax(np.abs(np.diff(prof)))
            fronts.append(z[front_idx])
        fronts_arr = np.array(fronts, dtype=float)

        #plot heat map of Tm at 1,2,2.5 ns:
        for t_plot in [1e-9, 2e-9, 2.5e-9]:
            idx_plot = np.argmin(np.abs(stored_t - t_plot))
            plt.imshow(stored_Tm[idx_plot], extent=(0.0, R, 0.0, Lz), origin='lower', aspect='auto')
            plt.colorbar(label='Material Temperature (HeV)')
            plt.xlabel('r (cm)')
            plt.ylabel('z (cm)')
            plt.title(f'Material Temperature Distribution at t = {t_plot*1e9} ns')
            plt.show()

        #plot front vs time
        plt.plot(stored_t * 1e9, fronts_arr * 1e1)  # time in ns, front in millimeters

        df = pd.read_csv("Project 1/data_articles_1_2/article_1_fronts/HR_simple.csv")
        # Adjust column names if needed
        t_csv = df["x"].to_numpy()
        x_csv = df["y"].to_numpy()
        plt.plot(t_csv, x_csv, linestyle="--", label="T_D (French)", color='red')

        df = pd.read_csv("Project 1/data_articles_1_2/article_1_fronts/HR_eff_1D.csv")
        # Adjust column names if needed
        t_csv = df["x"].to_numpy()
        x_csv = df["y"].to_numpy()
        plt.plot(t_csv, x_csv, linestyle="--", label="T_D (French)", color='black')

        plt.xlabel('Time (ns)')
        plt.ylabel('Front Position (millimeters)')
        plt.title('Front Position vs Time at r=0')
        plt.ylim(0, Lz*1e1)
        plt.grid()
        plt.show()

        

