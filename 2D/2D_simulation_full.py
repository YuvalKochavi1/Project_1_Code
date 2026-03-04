"""
Full 2D (z,r) diffusion solver (implicit, backward Euler) - a foam cylender coated with gold, with a drive at z=0. The material model is currently set to the foam, 
but can be adapted to other Material blocks by changing the material model hooks (sigma_of_T, beta_of_T, D_of_T, U_m_of_UR). The r=R boundary is currently set to 
Dirichlet with the same temperature as z=Lz (300 K), but can be switched to Neumann (dE/dr=0) by changing the bc_r_outer argument in the run() method. 
-----------------------------------------------------------------
This is the 2D generalization of your 1D self-similar implicit scheme:

- Variable diffusion coefficient: D(T) = c/(3*sigma(T))
- Face diffusion uses harmonic/arithmetic/geometric average (like your Eq. 23 → 2D)
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
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres, bicgstab, spilu, LinearOperator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["serif"]  # safe default SanSerif
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["mathtext.default"] = "regular"


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
        Lz, gold_width, R_foam, Nz, Nr_foam,
        # time
        dt_init, t_final,
        # unit system
        simulation_unit_system="cgs",  # "cgs" or "hev|ns"
        # material params (your self-similar)
        foam_params, gold_params,
        chi=1000.0,
        # drive
        t_drive_ns=None, T_drive_eV=None,
        # face averaging
        kind_of_D_face="arithmetic",  # harmonic/arithmetic/geometric
        # initial material temperature
        T_material_0_K=300.0,
        # linear solve controls (runtime-critical)
        linear_solver="direct",  # "direct" (matches reference) or "bicgstab" (faster, approximate)
        linear_tol=1e-10, # for iterative solver; ignored for direct: means "solve until convergence to 1e-10 residual" (relative)
        linear_maxiter=300, # for iterative solver; ignored for direct: maximum iterations before giving up and falling back to direct
        linear_check_residual=True, # for iterative solver; ignored for direct: whether to check residual after solve and fall back to direct if it's too large (recommended, especially for early iterations when solution is changing a lot)
        linear_residual_factor=50.0, # for iterative solver; ignored for direct: factor above which to fall back to direct solve (e.g. if linear_residual_factor=10, then if residual is >10 times initial residual, we consider it a failure and fall back to direct)
    ):
        self.Lz, self.gold_width = float(Lz), float(gold_width)
        self.R_foam = float(R_foam)
        self.Nz, self.Nr_foam = int(Nz), int(Nr_foam)
        self.z = np.linspace(0.0, self.Lz, self.Nz)
        self.r, self.r_info = make_r_two_block(self.R_foam, self.gold_width, self.Nr_foam, Nr_gold=30, dr0= self.gold_width/3000)
        self.Nr = self.r.size
        print(self.r)
        self.dz = self.z[1] - self.z[0]

        self.simulation_unit_system = simulation_unit_system
        self.kind_of_D_face = kind_of_D_face

        # Radial material maps (shape: (Nr,)). These broadcast naturally against (Nz, Nr) fields.
        mask_foam = (self.r < self.R_foam)
        self.f_map       = np.where(mask_foam, foam_params["f"],       gold_params["f"])
        self.g_map       = np.where(mask_foam, foam_params["g"],       gold_params["g"])
        self.alpha_map   = np.where(mask_foam, foam_params["alpha"],   gold_params["alpha"])
        self.betaexp_map = np.where(mask_foam, foam_params["beta_exp"],gold_params["beta_exp"])
        self.lam_map     = np.where(mask_foam, foam_params["lambda_param"], gold_params["lambda_param"])
        self.mu_map      = np.where(mask_foam, foam_params["mu"],      gold_params["mu"])
        self.rho_map     = np.where(mask_foam, foam_params["rho"],     gold_params["rho"])
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

        # --- runtime caches (iterative solver) ---
        self.linear_solver = str(linear_solver)
        self.linear_tol = float(linear_tol)
        self.linear_maxiter = int(linear_maxiter)
        self.linear_check_residual = bool(linear_check_residual)
        self.linear_residual_factor = float(linear_residual_factor)

        # CSR sparsity template is constant in time (only values change)
        self._csr_template = {}  # keyed by marshak_boundary bool

        # Precompute radial geometry factors for r-diffusion (independent of time)
        self._r_weights = self._precompute_r_weights(self.r)

    @staticmethod # staticmethod since it doesn't use self and can be useful on its own
    def _precompute_r_weights(r):
        r = np.asarray(r, dtype=float)
        Nr = r.size
        eps_r = 1e-30

        weights = {
            "eps_r": eps_r,
            "w_mh": np.zeros(Nr, dtype=float),
            "w_ph": np.zeros(Nr, dtype=float),
            "w_axis": 0.0,
            "w_mh_outer": 0.0,
            "w_ph_outer": 0.0,
        }

        if Nr <= 1:
            return weights

        # axis (j=0)
        r_ph = 0.5 * (r[0] + r[1])
        dr_ph = r[1] - r[0]
        dr_cv = r_ph - 0.0
        weights["w_axis"] = (r_ph) / ((r[0] + eps_r) * dr_cv * dr_ph)

        # interior (j=1..Nr-2)
        if Nr > 2:
            j = np.arange(1, Nr - 1)
            rj = r[j]
            r_mh = 0.5 * (r[j - 1] + r[j]) # mh = "minus half" since it's between j and j-1
            r_ph = 0.5 * (r[j] + r[j + 1]) # ph = "plus half" since it's between j and j+1
            dr_mh = r[j] - r[j - 1]
            dr_ph = r[j + 1] - r[j]
            dr_cv = r_ph - r_mh
            weights["w_mh"][j] = (r_mh) / ((rj + eps_r) * dr_cv * dr_mh)
            weights["w_ph"][j] = (r_ph) / ((rj + eps_r) * dr_cv * dr_ph)

        # outer boundary (j=Nr-1)
        j = Nr - 1
        rj = r[j]
        r_mh = 0.5 * (r[j - 1] + r[j])
        dr_mh = r[j] - r[j - 1]
        r_ph = r[j] + 0.5 * (r[j] - r[j - 1])
        dr_ph = r[j] - r[j - 1]
        dr_cv = r_ph - r_mh
        weights["w_mh_outer"] = (r_mh) / ((rj + eps_r) * dr_cv * dr_mh)
        weights["w_ph_outer"] = (r_ph) / ((rj + eps_r) * dr_cv * dr_ph)
        return weights

    def _ensure_csr_template(self, marshak_boundary: bool):
        """Builds and caches the CSR sparsity pattern for the linear system matrix. The pattern depends on whether the Marshak boundary condition is used at z=0, since that row has a different structure."""
        key = bool(marshak_boundary)
        if key in self._csr_template: # already built
            return

        Nz, Nr = self.Nz, self.Nr
        if key:
            i0, i1 = 0, Nz - 2
        else:
            i0, i1 = 1, Nz - 2
        nzi = (i1 - i0 + 1)
        n_unknown = nzi * Nr

        def idx(i, j):
            return (i - i0) * Nr + j

        indptr = np.zeros(n_unknown + 1, dtype=np.int64) # CSR indptr array
        indices_list = [] 
        #columns: indices[indptr[k]:indptr[k+1]] are the column indices for row k in the CSR matrix. We build this list of column indices for each row, and then convert to numpy array at the end.
        #values: data[indptr[k]:indptr[k+1]] are the corresponding values for row k. We initialize this to zeros and fill in the values during the solve phase, since the sparsity pattern is constant but the values change each iteration.

        # First pass: build row -> sorted column list
        for i in range(i0, i1 + 1):
            for j in range(Nr):
                row = idx(i, j)
                if key and i == 0:
                    cols = [row, idx(1, j)]
                else:
                    #The neighbors are always: self, i-1, i+1, j-1, j+1 (if they exist within bounds). We add them to the column list for this row if they are valid neighbors. We sort and deduplicate the column list at the end to ensure correct CSR format.
                    cols = [row]
                    if i > i0:
                        cols.append(idx(i - 1, j))
                    if i < i1:
                        cols.append(idx(i + 1, j))
                    if j > 0:
                        cols.append(idx(i, j - 1))
                    if j < Nr - 1:
                        cols.append(idx(i, j + 1))
                cols = sorted(set(cols))
                indices_list.extend(cols)
                indptr[row + 1] = indptr[row] + len(cols)

        indices = np.asarray(indices_list, dtype=np.int64)
        data = np.zeros(indices.size, dtype=np.float64)

        # Second pass: for each row, find positions of self and neighbors in CSR data
        pos_self = -np.ones(n_unknown, dtype=np.int64)
        pos_im = -np.ones(n_unknown, dtype=np.int64)
        pos_ip = -np.ones(n_unknown, dtype=np.int64)
        pos_jm = -np.ones(n_unknown, dtype=np.int64)
        pos_jp = -np.ones(n_unknown, dtype=np.int64)

        for i in range(i0, i1 + 1):
            for j in range(Nr):
                row = idx(i, j)
                start, end = indptr[row], indptr[row + 1]
                row_cols = indices[start:end]

                # always has diagonal
                pos_self[row] = start + int(np.searchsorted(row_cols, row))

                if key and i == 0:
                    col_ip = idx(1, j)
                    pos_ip[row] = start + int(np.searchsorted(row_cols, col_ip))
                    continue

                if i > i0:
                    col_im = idx(i - 1, j)
                    pos_im[row] = start + int(np.searchsorted(row_cols, col_im))
                if i < i1:
                    col_ip = idx(i + 1, j)
                    pos_ip[row] = start + int(np.searchsorted(row_cols, col_ip))
                if j > 0:
                    col_jm = idx(i, j - 1)
                    pos_jm[row] = start + int(np.searchsorted(row_cols, col_jm))
                if j < Nr - 1:
                    col_jp = idx(i, j + 1)
                    pos_jp[row] = start + int(np.searchsorted(row_cols, col_jp))

        self._csr_template[key] = {
            "i0": i0,
            "i1": i1,
            "nzi": nzi,
            "n_unknown": n_unknown,
            "indptr": indptr,
            "indices": indices,
            "data": data,
            "pos_self": pos_self,
            "pos_im": pos_im,
            "pos_ip": pos_ip,
            "pos_jm": pos_jm,
            "pos_jp": pos_jp,
        }

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
            return 1.0 / (self.g_map* (T_Hev ** self.alpha_map) * (self.rho_map** (-self.lam_map - 1)))
        else:
            return 1.0 / (self.g_map* (T ** self.alpha_map) * (self.rho_map** (-self.lam_map - 1)))
        
    def beta_of_T(self, T):
        """
        beta(T) = Cv_R / Cv_m with your conventions.
        Uses your same CGS/HeV logic.
        """
        if self.simulation_unit_system == "cgs":
            Cv_m = self.f_map* self.betaexp_map* (T ** (self.betaexp_map- 1)) * (self.rho_map** (-self.mu_map + 1))
            Cv_R = 4.0 * self.a * (T ** 3)
            return (Cv_R / Cv_m) * (K_per_Hev ** self.betaexp_map)
        else:
            return ((4.0 * self.a * (self.rho_map** (self.mu_map - 1))) / (self.f_map* self.betaexp_map)) * (T ** (4.0 - self.betaexp_map))

    def D_of_T(self, T):
        return self.c / (3.0 * self.sigma_of_T(T))

    def U_m_of_UR(self, UR):
        # Used only for diagnostics/energy integrals
        T = (UR / self.a) ** 0.25
        if self.simulation_unit_system == "cgs":
            T_Hev = T / K_per_Hev
        else:
            T_Hev = T
        return self.f_map* (T_Hev ** self.betaexp_map) * (self.rho_map** (-self.mu_map + 1))

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
        dz = self.dz

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

        # Build CSR matrix values efficiently (structure is cached)
        self._ensure_csr_template(marshak_boundary)
        tpl = self._csr_template[bool(marshak_boundary)]
        i0, i1 = tpl["i0"], tpl["i1"]
        nzi, n_unknown = tpl["nzi"], tpl["n_unknown"]
        indptr, indices = tpl["indptr"], tpl["indices"]
        pos_self = tpl["pos_self"] # position of diagonal entry for each row
        pos_im = tpl["pos_im"] # position of i-1 neighbor for each row (or -1 if no such neighbor)
        pos_ip = tpl["pos_ip"] # position of i+1 neighbor for each row (or -1 if no such neighbor)
        pos_jm = tpl["pos_jm"] # position of j-1 neighbor for each row (or -1 if no such neighbor)
        pos_jp = tpl["pos_jp"] # position of j+1 neighbor for each row (or -1 if no such neighbor)

        data = np.zeros_like(tpl["data"])
        b = np.zeros(n_unknown, dtype=np.float64)

        E_left = self.E_left_drive(t)
        E_right = self.E_right_bath()

        w_mh = self._r_weights["w_mh"]
        w_ph = self._r_weights["w_ph"]
        w_axis = self._r_weights["w_axis"]
        w_mh_outer = self._r_weights["w_mh_outer"]
        w_ph_outer = self._r_weights["w_ph_outer"]

        inv_dt = 1.0 / dt_local
        dz2 = dz * dz

        for i in range(i0, i1 + 1):
            base_row = (i - i0) * Nr
            rows = base_row + np.arange(Nr)

            if marshak_boundary and i == 0:
                alpha_vec = 2.0 * Dz_face[0, :] / (self.c * dz)
                data[pos_self[rows]] = 1.0 + alpha_vec
                data[pos_ip[rows]] = -alpha_vec
                b[rows] = self.E_left_drive(t + dt_local)
                continue

            diag = inv_dt + coupling[i, :]
            rhs = E_old[i, :] * inv_dt + coupling[i, :] * UR_old[i, :]

            # z diffusion
            D_imh = Dz_face[i - 1, :]
            D_iph = Dz_face[i, :]
            diag += (D_imh + D_iph) / dz2

            if i < i1:
                data[pos_ip[rows]] = -D_iph / dz2
            else:
                rhs += (D_iph / dz2) * E_right

            if i > i0:
                data[pos_im[rows]] = -D_imh / dz2
            else:
                if not marshak_boundary:
                    rhs += (D_imh / dz2) * E_left

            # r diffusion (nonuniform cylindrical)
            if Nr > 1:
                Dr_i = Dr_face[i, :]

                # axis j=0
                coeff0 = w_axis * Dr_i[0]
                diag[0] += coeff0
                data[pos_jp[rows[0]]] = -coeff0

                # interior j=1..Nr-2
                if Nr > 2:
                    j = np.arange(1, Nr - 1)
                    coeff_mh = w_mh[j] * Dr_i[j - 1]
                    coeff_ph = w_ph[j] * Dr_i[j]
                    diag[j] += coeff_mh + coeff_ph
                    data[pos_jm[rows[j]]] = -coeff_mh
                    data[pos_jp[rows[j]]] = -coeff_ph

                # outer j=Nr-1
                coeff_mh_o = w_mh_outer * Dr_i[-1]
                diag[-1] += coeff_mh_o
                data[pos_jm[rows[-1]]] = -coeff_mh_o
                if bc_r_outer == "dirichlet_bath":
                    coeff_ph_o = w_ph_outer * Dr_i[-1]
                    diag[-1] += coeff_ph_o
                    rhs[-1] += coeff_ph_o * E_right
                elif bc_r_outer == "neumann0":
                    pass
                else:
                    raise ValueError("bc_r_outer must be 'neumann0' or 'dirichlet_bath'.")

            data[pos_self[rows]] = diag
            b[rows] = rhs

        A_mat = csr_matrix((data, indices, indptr), shape=(n_unknown, n_unknown))

        # Warm start (helps iterative solvers a lot in time loops)
        x0 = E_old[i0 : i1 + 1, :].reshape(n_unknown)

        # Solve
        if self.linear_solver == "direct":
            E_inner = spsolve(A_mat, b)
        else:
            # diagonal (Jacobi) preconditioner: cheap and surprisingly effective here
            diag_entries = data[pos_self]
            inv_diag = 1.0 / (diag_entries + 1e-300)

            def precond(v):
                return inv_diag * v

            M = LinearOperator((n_unknown, n_unknown), matvec=precond, dtype=np.float64)
            try:
                # SciPy >= 1.8 typically uses rtol/atol
                E_inner, info = bicgstab(
                    A_mat,
                    b,
                    x0=x0,
                    rtol=self.linear_tol,
                    atol=0.0,
                    maxiter=self.linear_maxiter,
                    M=M,
                )
            except TypeError:
                # Older SciPy uses tol
                E_inner, info = bicgstab(
                    A_mat,
                    b,
                    x0=x0,
                    tol=self.linear_tol,
                    maxiter=self.linear_maxiter,
                    M=M,
                )
            need_fallback = (info != 0) or (not np.all(np.isfinite(E_inner)))
            if (not need_fallback) and self.linear_check_residual:
                # Guard against "converged" but inaccurate solutions
                resid = A_mat @ E_inner - b
                rel_resid = np.linalg.norm(resid) / (np.linalg.norm(b) + 1e-30)
                if rel_resid > self.linear_residual_factor * self.linear_tol:
                    need_fallback = True

            if need_fallback:
                E_inner = spsolve(A_mat, b)

        # reconstruct full E^{n+1} without Python loops
        E_new = E_old.copy()
        E_new[i0 : i1 + 1, :] = E_inner.reshape((nzi, Nr))
        if not marshak_boundary:
            E_new[0, :] = E_left
        E_new[-1, :] = E_right

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

    def compute_front_at_r(
        self,
        stored_Tm,
        *,
        r_index: int = 0,
        front_method: str = "maxgrad",
        threshold: float = 5,
        T_cold=None,
    ):
        """Compute front position z_F(t) for a single radial index.

        Parameters
        ----------
        stored_Tm:
            Array of shape (Nt, Nz, Nr) (as returned by :meth:`run`) or a single snapshot (Nz, Nr).

        r_index:
            Radial index j to evaluate the front on.

        front_method:
            - "maxgrad": z_F = argmax_z |dT/dz|.
            - "threshold": z_F = first z where T <= threshold*T_bath.

        Returns
        -------
        zF_cm: (Nt,)
            Front location in cm along z for each stored time.
        """

        Tm = np.asarray(stored_Tm)
        if Tm.ndim == 2:
            Tm = Tm[None, :, :] # add time axis if missing
        if Tm.ndim != 3:
            raise ValueError("stored_Tm must have shape (Nt, Nz, Nr) or (Nz, Nr).")
        if Tm.shape[1] != self.Nz or Tm.shape[2] != self.Nr:
            raise ValueError(
                f"Expected (Nt, Nz, Nr)=(*, {self.Nz}, {self.Nr}) but got {Tm.shape}."
            )

        r_index = int(r_index)
        if not (0 <= r_index < self.Nr):
            raise ValueError(f"r_index must be in [0, {self.Nr-1}]")

        method = str(front_method).strip().lower()
        if method not in {"maxgrad", "threshold"}:
            raise ValueError("front_method must be 'maxgrad' or 'threshold'.")

        prof = Tm[:, :, r_index]  # (Nt, Nz)

        if method == "maxgrad":
            dT = np.abs(np.diff(prof, axis=1))  # (Nt, Nz-1)
            idx = np.argmax(dT, axis=1)  # (Nt,)
            return np.take(self.z, idx)

        # threshold method
        if T_cold is None:
            if self.simulation_unit_system == "cgs":
                T_cold = 300.0
            else:
                T_cold = 300.0 / K_per_Hev
        T_cold = float(T_cold)
        threshold = float(threshold)

        mask = prof <= (threshold * T_cold)  # (Nt, Nz)
        idx = np.argmax(mask, axis=1)  # (Nt,) (0 if all-False)
        none = ~np.any(mask, axis=1)
        if np.any(none):
            idx = idx.copy()
            idx[none] = self.Nz - 1
        return np.take(self.z, idx)

    def _compute_energy_region(self, stored_Um, *, mask_r):
        """Internal: axisymmetric energy integral over a radial mask.

        Returns total energy in erg for each stored time.
        """

        Um = np.asarray(stored_Um)
        if Um.ndim == 2:
            Um = Um[None, :, :]
        if Um.ndim != 3:
            raise ValueError("stored_Um must have shape (Nt, Nz, Nr) or (Nz, Nr).")
        if Um.shape[1] != self.Nz or Um.shape[2] != self.Nr:
            raise ValueError(
                f"Expected (Nt, Nz, Nr)=(*, {self.Nz}, {self.Nr}) but got {Um.shape}."
            )

        r_nodes = np.asarray(self.r, dtype=float)
        mask_r = np.asarray(mask_r, dtype=bool)
        if mask_r.shape != (self.Nr,):
            raise ValueError(f"mask_r must have shape ({self.Nr},)")
        if np.count_nonzero(mask_r) < 2:
            raise ValueError("Energy region mask selects <2 radial nodes; cannot integrate.")

        r_int = r_nodes[mask_r]
        weight_r = 2.0 * np.pi * r_int  # (Nr_int,)
        z = self.z

        energies_erg = []
        for Ui in Um:
            Ui_int = Ui[:, mask_r]  # (Nz, Nr_int)
            integrand = Ui_int * weight_r[None, :]
            ez = np.trapezoid(integrand, r_int, axis=1)  # (Nz,)
            energies_erg.append(float(np.trapezoid(ez, z, axis=0)))

        return np.asarray(energies_erg)

    def compute_energy_foam(self, stored_Um):
        """Total foam energy vs time (axisymmetric), in erg.

        Foam region is defined consistently with the material mask: r < R_foam.
        Convert to hJ by multiplying by 1e-9 (since 1 erg = 1e-9 hJ).
        """

        r_nodes = np.asarray(self.r, dtype=float)
        mask_foam = r_nodes < float(self.R_foam)
        return self._compute_energy_region(stored_Um, mask_r=mask_foam)

    def compute_energy_gold(self, stored_Um):
        """Total gold energy vs time (axisymmetric), in erg.

        Gold region is defined consistently with the material mask: r >= R_foam.
        Convert to hJ by multiplying by 1e-9 (since 1 erg = 1e-9 hJ).
        """

        r_nodes = np.asarray(self.r, dtype=float)
        mask_gold = r_nodes >= float(self.R_foam)
        return self._compute_energy_region(stored_Um, mask_r=mask_gold)

    def compute_front_surface(
        self,
        stored_Tm,
        *,
        front_method: str = "maxgrad",
        threshold: float = 5,
        T_cold=None,
    ):
        """Compute the front surface z_F(r,t) from stored 2D material temperature.

        Parameters
        ----------
        stored_Tm:
            Array of shape (Nt, Nz, Nr) (as returned by :meth:`run`) or a single snapshot (Nz, Nr).

        front_method:
            - "maxgrad": for each r and t, front is argmax_z |dT/dz|.
            - "threshold": for each r and t, front is first z where T <= threshold*T_cold.

        threshold, T_cold:
            Used only for front_method="threshold".
            By default, T_cold is 300 K (CGS) or 300/K_per_Hev (hev|ns).

        Returns
        -------
        zF_cm:
            Array of shape (Nt, Nr) giving the front location in cm for each (t, r).
        """

        Tm = np.asarray(stored_Tm)
        if Tm.ndim == 2:
            Tm = Tm[None, :, :]

        if Tm.ndim != 3:
            raise ValueError("stored_Tm must have shape (Nt, Nz, Nr) or (Nz, Nr).")
        if Tm.shape[1] != self.Nz or Tm.shape[2] != self.Nr:
            raise ValueError(
                f"Expected (Nt, Nz, Nr)=(*, {self.Nz}, {self.Nr}) but got {Tm.shape}."
            )

        method = str(front_method).strip().lower()
        if method not in {"maxgrad", "threshold"}:
            raise ValueError("front_method must be 'maxgrad' or 'threshold'.")

        if method == "maxgrad":
            dT = np.abs(np.diff(Tm, axis=1))  # (Nt, Nz-1, Nr)
            idx = np.argmax(dT, axis=1)  # (Nt, Nr)
            return np.take(self.z, idx)

        # threshold method
        if T_cold is None:
            if self.simulation_unit_system == "cgs":
                T_cold = 300.0
            else:
                T_cold = 300.0 / K_per_Hev
        T_cold = float(T_cold)
        threshold = float(threshold)

        mask = Tm <= (threshold * T_cold)  # (Nt, Nz, Nr)
        idx = np.argmax(mask, axis=1)  # (Nt, Nr) (0 if all-False)
        none = ~np.any(mask, axis=1)
        if np.any(none):
            idx = idx.copy()
            idx[none] = self.Nz - 1
        return np.take(self.z, idx)

    
# ============================================================
# a geometric changing grid in r in the gold region
# ============================================================
# find q such that dr0 + dr0*q + dr0*q^2 + ... + dr0*q^(N-1) =~ gold_width
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

    # uniform special case
    if abs(dr0 * N - gold_width) / gold_width < 1e-12:
        return 1.0

    def S(q):
        return dr0 * (q**N - 1.0) / (q - 1.0)

    q_lo = 1.0 + 1e-12
    q_hi = 2.0
    # The following loop finds an upper bound q_hi such that S(q_hi) >= gold_width, 
    # starting from q_lo=1.0 (uniform) and doubling until we exceed gold_width.
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
      - foam region [0, R_foam): uniform with Nr_foam nodes (endpoint=False)
      - gold region [R_foam, R_foam+gold_width]: geometric widths

    Choose dr0 = first gold cell width at the foam–gold interface (I can choose this to be small, e.g. gold_width/1000)

    Returns:
      r      : 1D array of nodes from 0 to R_foam+gold_width
      info   : dict with q, dr0, widths, R_total
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

    # Gold widths
    q = solve_q_from_dr0(gold_width, Nr_gold, dr0)
    widths = dr0 * (q ** np.arange(Nr_gold))

    # Gold nodes (include R_foam and R_total)
    r_gold = R_foam + np.concatenate(([0.0], np.cumsum(widths)))

    # Enforce exact outer radius (avoid floating drift)
    R_total = R_foam + gold_width
    r_gold[-1] = R_total

    # Merge (drop duplicate R_foam)
    r = np.concatenate((r_foam, r_gold))   # keep R_foam node
    # (and remove duplicates safely if you want)
    r = np.unique(r)

    info = {"q": float(q), "dr0": float(dr0), "widths": widths, "R_total": R_total}
    return r, info


def edges_from_nodes_with_bounds(x, x_left, x_right):
    x = np.asarray(x, dtype=float)
    xe = np.empty(x.size + 1, dtype=float)
    xe[1:-1] = 0.5 * (x[:-1] + x[1:])
    xe[0] = float(x_left)
    xe[-1] = float(x_right)
    return xe


def cell_to_vertices(T_cell):
    """Convert cell-centered (Nz, Nr) array to vertex-centered (Nz+1, Nr+1).

    Useful for pcolormesh with shading='gouraud'.
    """

    T_cell = np.asarray(T_cell)
    Nz, Nr = T_cell.shape
    Tv = np.empty((Nz + 1, Nr + 1), dtype=T_cell.dtype)

    # interior vertices: 4-cell average
    Tv[1:Nz, 1:Nr] = 0.25 * (
        T_cell[:-1, :-1] + T_cell[1:, :-1] + T_cell[:-1, 1:] + T_cell[1:, 1:]
    )

    # edges: copy nearest cell row/col
    Tv[0, 1:Nr] = T_cell[0, :-1]  # z=0
    Tv[Nz, 1:Nr] = T_cell[-1, :-1]  # z=Lz
    Tv[1:Nz, 0] = T_cell[:-1, 0]  # r=0
    Tv[1:Nz, Nr] = T_cell[:-1, -1]  # r=R

    # corners
    Tv[0, 0] = T_cell[0, 0]
    Tv[0, Nr] = T_cell[0, -1]
    Tv[Nz, 0] = T_cell[-1, 0]
    Tv[Nz, Nr] = T_cell[-1, -1]
    return Tv


def create_simulation(
    *,
    material: str = "SiO2",
    Nz: int = 400,
    Nr_foam: int = 400,
    kind_of_D_face: str = "arithmetic",
    chi: float = 1000.0,
    T_material_0_K: float = 300.0,
):
    """Factory: build a configured :class:`SelfSimilarDiffusion2D`.

    This extracts the parameter setup from the old __main__ block so it can be
    imported and called from other scripts.
    """

    material = str(material)
    if material == "SiO2":
        # self similarity model fudge factors - Foam
        f = 8.77 * 10**13
        g = 1 / 9175
        alpha = 3.53
        beta_exp = 1.1
        lambda_param = 0.75
        mu = 0.09
        rho = 0.05

        # Geometry / grid
        Lz = 0.3  # cm
        R_foam = 0.08
        gold_width = 25 * 1e-4  # 25 microns in cm
        csv_path = "Project 1/data_articles_1_2/article_1_tempertures/T_drive.csv"
        t_drive_ns, T_drive_eV = load_time_temp(csv_path)

        # Time
        t_final = 3e-9
        dt_init = 5e-15
    else:
        raise ValueError(f"{material} is not supported in this function for now.")

    foam_params = {
        "f": f,
        "g": g,
        "alpha": alpha,
        "beta_exp": beta_exp,
        "lambda_param": lambda_param,
        "mu": mu,
        "rho": rho,
    }
    gold_params = {
        "f": 3.4e13,
        "g": 1 / 7200,
        "alpha": 1.5,
        "beta_exp": 1.6,
        "lambda_param": 0.2,
        "mu": 0.14,
        "rho": 19.32,
    }

    return SelfSimilarDiffusion2D(
        Lz=Lz,
        gold_width=gold_width,
        R_foam=R_foam,
        Nz=int(Nz),
        Nr_foam=int(Nr_foam),
        dt_init=dt_init,
        t_final=t_final,
        simulation_unit_system="cgs",
        foam_params=foam_params,
        gold_params=gold_params,
        chi=float(chi),
        t_drive_ns=t_drive_ns,
        T_drive_eV=T_drive_eV,
        kind_of_D_face=str(kind_of_D_face),
        T_material_0_K=float(T_material_0_K),
    )


def run_simulation(
    sim: "SelfSimilarDiffusion2D",
    *,
    n_store: int = 50,
    store_start_frac: float = 0.01,
    dtfac: float = 0.05,
    dtmin: float | None = 5e-15,
    dtmax: float | None = 2e-12,
    bc_r_outer: str = "dirichlet_bath",
    marshak_boundary: bool = True,
):
    """Run a simulation and return stored arrays (same as the old script)."""

    times_to_store = sim.t_final * np.linspace(float(store_start_frac), 1.0, int(n_store))
    stored_t, stored_Um, stored_Tm, stored_TR = sim.run(
        times_to_store,
        dtfac=float(dtfac),
        dtmin=dtmin,
        dtmax=dtmax,
        bc_r_outer=str(bc_r_outer),
        marshak_boundary=bool(marshak_boundary),
    )
    return stored_t, stored_Um, stored_Tm, stored_TR


def save_run_data(file_path, stored_t, stored_Um=None, stored_Tm=None, stored_TR=None):
    """Save (stored_t, stored_Um, stored_Tm, stored_TR) to a single .npz file.

    You can call either:
      - save_run_data(path, stored_t, stored_Um, stored_Tm, stored_TR)
      - save_run_data(path, (stored_t, stored_Um, stored_Tm, stored_TR))
    """

    if stored_Um is None and stored_Tm is None and stored_TR is None:
        stored_t, stored_Um, stored_Tm, stored_TR = stored_t

    file_path = str(file_path)
    if not file_path.lower().endswith(".npz"):
        file_path += ".npz"

    out_dir = os.path.dirname(file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        file_path,
        stored_t=np.asarray(stored_t),
        stored_Um=np.asarray(stored_Um),
        stored_Tm=np.asarray(stored_Tm),
        stored_TR=np.asarray(stored_TR),
    )
    return file_path

def plot_temperature_maps_gouraud(
    sim: "SelfSimilarDiffusion2D",
    stored_t,
    stored_Tm,
    *,
    times_s=(1e-9, 2e-9, 2.5e-9),
    out_dir="Project 1/2D/figures",
):
    os.makedirs(out_dir, exist_ok=True)
    for t_plot in times_s:
        idx_plot = int(np.argmin(np.abs(stored_t - t_plot)))
        T_cell = stored_Tm[idx_plot] / K_per_Hev  # (Nz, Nr) in HeV

        R_total = sim.R_foam + sim.gold_width
        r_edges = edges_from_nodes_with_bounds(sim.r, 0.0, R_total)
        z_edges = edges_from_nodes_with_bounds(sim.z, 0.0, sim.Lz)
        T_vert = cell_to_vertices(T_cell)

        Re, Ze = np.meshgrid(r_edges, z_edges)

        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        pcm = ax.pcolormesh(Re, Ze, T_vert, shading="gouraud", cmap="Spectral_r")

        ax.axvline(sim.R_foam, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("r (cm)", fontname="serif")
        ax.set_ylabel("z (cm)", fontname="serif")
        ax.set_title(
            f"Material temperature T(r,z) at t = {t_plot*1e9:.1f} ns",
            fontname="serif",
        )
        ax.set_xlim(0.0, R_total)
        ax.set_ylim(0.0, sim.Lz)

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Material temperature T (HeV)", fontname="serif")

        fig.savefig(os.path.join(out_dir, f"Tmap_{t_plot*1e9:.1f}ns.png"), dpi=250)
        plt.close(fig)


def plot_temperature_maps_simple(
    sim: "SelfSimilarDiffusion2D",
    stored_t,
    stored_Tm,
    *,
    times_s=(1e-9, 2e-9, 2.5e-9),
    out_dir="Project 1/2D/figures",
):
    os.makedirs(out_dir, exist_ok=True)
    for t_plot in times_s:
        idx_plot = int(np.argmin(np.abs(stored_t - t_plot)))
        R_total = sim.R_foam + sim.gold_width
        r_edges = edges_from_nodes_with_bounds(sim.r, 0.0, R_total)
        z_edges = edges_from_nodes_with_bounds(sim.z, 0.0, sim.Lz)
        Tm = stored_Tm[idx_plot]

        plt.figure()
        plt.pcolormesh(r_edges, z_edges, Tm, shading="auto", cmap="Spectral_r")
        plt.axvline(sim.R_foam, linestyle="--")
        plt.xlim(0.0, R_total)
        plt.ylim(0.0, sim.Lz)
        plt.colorbar(label="Material Temperature")
        plt.title(f"Material Temperature Tm at t={t_plot*1e9:.1f} ns")
        plt.xlabel("r (cm)")
        plt.ylabel("z (cm)")
        plt.savefig(os.path.join(out_dir, f"heatmap_{t_plot*1e9:.1f}ns.png"))
        plt.close()


def plot_front_vs_time(
    stored_t,
    front_z_cm,
    *,
    out_path="Project 1/2D/figures/front_position - Front Position vs Time at r=0.png",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(stored_t * 1e9, np.asarray(front_z_cm) * 1e1)  # ns, mm

    overlays = [
        ("Project 1/data_articles_1_2/article_1_fronts/HR_simple.csv", "Simple HR", "--", "red", 1.0),
        (
            "Project 1/data_articles_1_2/article_1_fronts/HR_eff_1D.csv",
            "Marshak boundary",
            "--",
            "black",
            1.0,
        ),
        (
            "Project 1/2D/gold_supersonic/shay_model_first_sent.csv",
            "Gold wall - Supersonic (shay's model)",
            "--",
            "blue",
            1.0,
        ),
        (
            "Project 1/2D/gold_supersonic/shay_simulation.csv",
            "Gold wall - Supersonic (Avner's simulation)",
            "-",
            "blue",
            1.0,
        ),
        (
            "Project 1/2D/gold_supersonic/gold_wall_subsonic.csv",
            "Gold wall - Subsonic (my 1.5 model)",
            "-",
            "red",
            1.0,
        ),
        (
            "Project 1/2D/gold_supersonic/my1_5_model_supersonic.csv",
            "Gold wall - Supersonic (my 1.5 model)",
            "--",
            "red",
            10.0,
        ),
    ]

    for path, label, ls, color, yscale in overlays:
        df = pd.read_csv(path)
        t_csv = df["x"].to_numpy()
        x_csv = df["y"].to_numpy()
        plt.plot(t_csv, yscale * x_csv, linestyle=ls, label=label, color=color)

    plt.xlabel("Time (ns)", fontname="serif")
    plt.ylabel("Front Position (millimeters)", fontname="serif")
    plt.title("Front Position vs Time at r=0", fontname="serif")
    plt.ylim(0, 2)
    plt.grid()
    plt.legend(prop={"family": "serif"})
    plt.savefig(out_path)
    plt.close()


def plot_front_surface(
    sim: "SelfSimilarDiffusion2D",
    stored_t,
    stored_Tm,
    *,
    times_s=(1e-9, 2e-9, 2.5e-9),
    out_path="Project 1/2D/figures/front_surface - Front Surface zF vs r.png",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    R_total = sim.R_foam + sim.gold_width

    zF = sim.compute_front_surface(stored_Tm, front_method="maxgrad")  # (Nt, Nr)
    plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    for t_plot in times_s:
        idx_plot = int(np.argmin(np.abs(stored_t - t_plot)))
        plt.plot(sim.r, zF[idx_plot], label=f"t={t_plot*1e9:.1f} ns")
    plt.axvline(sim.R_foam, color="k", linestyle="--", linewidth=1, label="Foam–Gold Interface")
    plt.xlabel("r (cm)", fontname="serif")
    plt.ylabel("Front Position z_F (cm)", fontname="serif")
    plt.title("Front Surface z_F(r) at Different Times", fontname="serif")
    plt.xlim(0.0, R_total)
    plt.ylim(0.0, sim.Lz)
    plt.grid()
    plt.legend(prop={"family": "serif"})
    plt.savefig(out_path)
    plt.close()


def plot_energy_comparison(
    sim: "SelfSimilarDiffusion2D",
    stored_t,
    stored_Um,
    *,
    csv_gold_supersonic="Project 1/2D/gold_supersonic/supersonic_gold_lost_energy.csv",
    csv_gold_1D="Project 1/data_articles_1_2/article_1_energies/total_energy_1D.csv",
    csv_gold_2D="Project 1/data_articles_1_2/article_1_energies/total_energy_2D.csv",
    out_path="Project 1/2D/figures/energy_comparison - Foam Energy vs Time.png",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    energy_foam = sim.compute_energy_foam(stored_Um) * 1e-9  # erg -> hJ
    energy_gold = sim.compute_energy_gold(stored_Um) * 1e-9  # erg -> hJ

    plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    plt.plot(stored_t * 1e9, energy_foam, label="Simulated Foam Energy (hJ)", color="blue")
    plt.plot(stored_t * 1e9, energy_gold, label="Simulated Gold Energy (hJ)", color="green")

    df = pd.read_csv(csv_gold_supersonic)
    t_csv = df["x"].to_numpy()
    energy_csv = df["y"].to_numpy()
    plt.plot(
        t_csv,
        energy_csv,
        linestyle="--",
        label="Estimated Lost Energy (units from CSV)",
        color="red",
    )

    df = pd.read_csv(csv_gold_1D)
    t_csv = df["x"].to_numpy()
    energy_csv = df["y"].to_numpy()
    plt.plot(
        t_csv,
        energy_csv,
        linestyle="-.",
        label="Total Energy 1D (units from CSV)",
        color="orange",
    )

    df = pd.read_csv(csv_gold_2D)
    t_csv = df["x"].to_numpy()
    energy_csv = df["y"].to_numpy()
    plt.plot(
        t_csv,
        energy_csv,
        linestyle="--",
        label="Total Energy 2D (units from CSV)",
        color="purple",
    )

    plt.xlabel("Time (ns)", fontname="serif")
    plt.ylabel("Foam Energy (hJ)", fontname="serif")
    plt.title("Foam Energy vs Time", fontname="serif")
    plt.grid()
    plt.legend(prop={"family": "serif"})
    plt.savefig(out_path)
    plt.close()


def run_default_pipeline(*, material: str = "SiO2"):
    """Replicates the old __main__ behavior, but callable from outside."""

    sim = create_simulation(material=material)
    stored_t, stored_Um, stored_Tm, stored_TR = run_simulation(
        sim,
        n_store=50,
        store_start_frac=0.01,
        dtfac=0.05,
        dtmin=5e-15,
        dtmax=2e-12,
        bc_r_outer="dirichlet_bath",
        marshak_boundary=True,
    )

    # Front at r=0 (maxgrad in z)
    front_z_cm = sim.compute_front_at_r(stored_Tm, r_index=0, front_method="maxgrad")

    plot_temperature_maps_gouraud(sim, stored_t, stored_Tm)
    plot_temperature_maps_simple(sim, stored_t, stored_Tm)
    plot_front_vs_time(stored_t, front_z_cm)
    plot_front_surface(sim, stored_t, stored_Tm)
    plot_energy_comparison(sim, stored_t, stored_Um)

    return sim, stored_t, stored_Um, stored_Tm, stored_TR


if __name__ == "__main__":
    run_default_pipeline(material="SiO2")
