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
import numpy as np
import tqdm
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve, bicgstab, LinearOperator  # type: ignore[import-untyped]

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


def outer_dr(r):
    """Return the last radial spacing used at the outer boundary face."""
    r = np.asarray(r, dtype=float)
    if r.size < 2:
        return 1.0
    return float(r[-1] - r[-2])

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
        foam_params, gold_params, be_params,
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
        self.r, self.r_info = make_r_two_block(self.R_foam, self.gold_width, self.Nr_foam, Nr_gold=35, dr0= self.gold_width/3000)
        self.Nr = self.r.size
        print(self.r)
        self.dz = self.z[1] - self.z[0]

        self.simulation_unit_system = simulation_unit_system
        self.kind_of_D_face = kind_of_D_face

        # Radial material maps (shape: (Nr,)). These broadcast naturally against (Nz, Nr) fields.
        mask_foam = (self.r <= self.R_foam)
        self.f_map       = np.where(mask_foam, foam_params["f"],       gold_params["f"])
        self.g_map       = np.where(mask_foam, foam_params["g"],       gold_params["g"])
        self.alpha_map   = np.where(mask_foam, foam_params["alpha"],   gold_params["alpha"])
        self.betaexp_map = np.where(mask_foam, foam_params["beta_exp"],gold_params["beta_exp"])
        self.lam_map     = np.where(mask_foam, foam_params["lambda_param"], gold_params["lambda_param"])
        self.mu_map      = np.where(mask_foam, foam_params["mu"],      gold_params["mu"])
        self.rho_map     = np.where(mask_foam, foam_params["rho"],     gold_params["rho"])
        
        # self.f_map       = np.where(mask_foam, foam_params["f"],       be_params["f"])
        # self.g_map       = np.where(mask_foam, foam_params["g"],       be_params["g"])
        # self.alpha_map   = np.where(mask_foam, foam_params["alpha"],   be_params["alpha"])
        # self.betaexp_map = np.where(mask_foam, foam_params["beta_exp"],be_params["beta_exp"])
        # self.lam_map     = np.where(mask_foam, foam_params["lambda_param"], be_params["lambda_param"])
        # self.mu_map      = np.where(mask_foam, foam_params["mu"],      be_params["mu"])
        # self.rho_map     = np.where(mask_foam, foam_params["rho"],     be_params["rho"])
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
    # NEW: Helper methods for face-based diffusion calculation
    # ============================================================
    def _compute_T_face_z(self, T):
        """
        Compute face temperatures at z-interfaces (between z-layers).
        
        For each pair of adjacent cells (i, i+1), compute:
        T_face = ((T_i^4 + T_{i+1}^4) / 2)^(1/4)
        
        Parameters
        ----------
        T : ndarray, shape (Nz, Nr)
            Temperature field at cell centers.
        
        Returns
        -------
        T_z_face : ndarray, shape (Nz-1, Nr)
            Temperature at z-faces (between cells i and i+1).
        """
        T = np.asarray(T, dtype=float)
        if T.shape != (self.Nz, self.Nr):
            raise ValueError(
                f"Expected T shape ({self.Nz}, {self.Nr}) for z-face temperature, got {T.shape}."
            )

        # Adjacent z cells: (i, j) and (i+1, j)
        T_i = T[:-1, :]
        T_ip1 = T[1:, :]
        T_z_face = ((T_i ** 4 + T_ip1 ** 4) / 2.0) ** 0.25
        return T_z_face
    
    def _compute_T_face_r(self, T):
        """
        Compute face temperatures at r-interfaces (between r-layers).
        
        For each pair of adjacent cells (j, j+1), compute:
        T_face = ((T_j^4 + T_{j+1}^4) / 2)^(1/4)
        
        Parameters
        ----------
        T : ndarray, shape (Nz, Nr)
            Temperature field at cell centers.
        
        Returns
        -------
        T_r_face : ndarray, shape (Nz, Nr-1)
            Temperature at r-faces (between cells j and j+1).
        """
        T = np.asarray(T, dtype=float)
        if T.shape != (self.Nz, self.Nr):
            raise ValueError(
                f"Expected T shape ({self.Nz}, {self.Nr}) for r-face temperature, got {T.shape}."
            )

        # Adjacent r cells: (i, j) and (i, j+1)
        T_j = T[:, :-1]
        T_jp1 = T[:, 1:]
        T_r_face = ((T_j ** 4 + T_jp1 ** 4) / 2.0) ** 0.25
        return T_r_face
    
    def _is_material_interface_r(self):
        """
        Detect r-faces that lie at material boundaries (foam-gold interface).
        
        Returns
        -------
        interface_faces : ndarray, shape (Nr-1,), dtype=bool
            True where a face is at the material interface (foam <-> gold).
        """
        r_mask_foam = self.r < self.R_foam
        interface_faces = r_mask_foam[:-1] != r_mask_foam[1:]
        return interface_faces
    
    def _compute_D_at_face_with_material_interface(self, T_r_face, interface_faces_r):
        """
        Compute D at r-faces, accounting for material interfaces.
        
        For internal faces (same material on both sides):
            D = D(T_face)
        
        For interface faces (foam-gold boundary):
            D_foam = D(T_face; foam_params)
            D_gold = D(T_face; gold_params)
            D_interface = 2 / (1/D_foam + 1/D_gold)  [harmonic mean]
        
        Parameters
        ----------
        T_r_face : ndarray, shape (Nz, Nr-1)
            Temperature at r-faces.
        interface_faces_r : ndarray, shape (Nr-1,), dtype=bool
            True where a face is at the material interface (foam <-> gold).
        
        Returns
        -------
        D_r_face : ndarray, shape (Nz, Nr-1)
            Diffusion coefficient at r-faces.
        """
        Nz = T_r_face.shape[0]
        Nr_minus_1 = T_r_face.shape[1]
        D_r_face = np.zeros_like(T_r_face, dtype=float)
        
        # Loop over r-face indices
        for j in range(Nr_minus_1):
            T_face_j = T_r_face[:, j]  # Temperature at r-face j, shape (Nz,)
            
            if not interface_faces_r[j]:
                # Internal face: use material properties of cell j (or j+1, same material)
                # Compute D using the material map values for this r-index
                if self.simulation_unit_system == "cgs":
                    T_Hev = T_face_j / K_per_Hev
                    g_j = self.g_map[j]
                    alpha_j = self.alpha_map[j]
                    lam_j = self.lam_map[j]
                    rho_j = self.rho_map[j]
                    sigma_j = 1.0 / (g_j * (T_Hev ** alpha_j) * (rho_j ** (-lam_j - 1)))
                    D_r_face[:, j] = self.c / (3.0 * sigma_j)
                else:  # hev|ns
                    T_hev = T_face_j
                    g_j = self.g_map[j]
                    alpha_j = self.alpha_map[j]
                    lam_j = self.lam_map[j]
                    rho_j = self.rho_map[j]
                    sigma_j = 1.0 / (g_j * (T_hev ** alpha_j) * (rho_j ** (-lam_j - 1)))
                    D_r_face[:, j] = self.c / (3.0 * sigma_j)
            else:
                # Interface face: compute D for both materials and take harmonic mean
                # Cell j is foam, cell j+1 is gold (based on interface detection)
                if self.simulation_unit_system == "cgs":
                    T_Hev = T_face_j / K_per_Hev
                    
                    # Foam side (cell j)
                    g_foam = self.g_map[j]
                    alpha_foam = self.alpha_map[j]
                    lam_foam = self.lam_map[j]
                    rho_foam = self.rho_map[j]
                    sigma_foam = 1.0 / (g_foam * (T_Hev ** alpha_foam) * (rho_foam ** (-lam_foam - 1)))
                    D_foam = self.c / (3.0 * sigma_foam)
                    
                    # Gold side (cell j+1)
                    g_gold = self.g_map[j + 1]
                    alpha_gold = self.alpha_map[j + 1]
                    lam_gold = self.lam_map[j + 1]
                    rho_gold = self.rho_map[j + 1]
                    sigma_gold = 1.0 / (g_gold * (T_Hev ** alpha_gold) * (rho_gold ** (-lam_gold - 1)))
                    D_gold = self.c / (3.0 * sigma_gold)
                else:  # hev|ns
                    T_hev = T_face_j
                    
                    # Foam side (cell j)
                    g_foam = self.g_map[j]
                    alpha_foam = self.alpha_map[j]
                    lam_foam = self.lam_map[j]
                    rho_foam = self.rho_map[j]
                    sigma_foam = 1.0 / (g_foam * (T_hev ** alpha_foam) * (rho_foam ** (-lam_foam - 1)))
                    D_foam = self.c / (3.0 * sigma_foam)
                    
                    # Gold side (cell j+1)
                    g_gold = self.g_map[j + 1]
                    alpha_gold = self.alpha_map[j + 1]
                    lam_gold = self.lam_map[j + 1]
                    rho_gold = self.rho_map[j + 1]
                    sigma_gold = 1.0 / (g_gold * (T_hev ** alpha_gold) * (rho_gold ** (-lam_gold - 1)))
                    D_gold = self.c / (3.0 * sigma_gold)
                # Harmonic mean: D_interface = 2 / (1/D_foam + 1/D_gold)
                eps = 1e-30
                #D_r_face[:, j] = 2 / (1.0 / (D_foam + eps) + 1.0 / (D_gold + eps) + eps)
                D_r_face[:, j] = 0.5 * (D_foam + D_gold)  # Alternatively, use arithmetic mean at interface for testing
        
        return D_r_face

    # ============================================================
    # Implicit step: build and solve sparse system for E^{n+1}
    # ============================================================
    def implicit_step(self, *, t, dt_local, bc_r_outer="marshak_wall", marshak_boundary=False):
        """
        bc_r_outer: "neumann0" (dE/dr=0 at r=R), "dirichlet_bath", or "marshak_wall"
        Axis r=0 always uses neumann symmetry.
        z=0: Dirichlet drive, z=Lz: Dirichlet bath
        """
        Nz, Nr = self.Nz, self.Nr
        dz = self.dz

        E_n = self.E  # E^n: radiation energy at time n
        U_n = self.UR  # U^n: material energy at time n

        # lagged coefficients from U^n
        T_n = (U_n / self.a) ** 0.25  # T^n: temperature at time n
        
        # # --- OLD IMPLEMENTATION (kept for reference) ---
        D_n = self.D_of_T(T_n)  # D^n at cell centers: diffusion coefficient at time n
        
        # # --- NEW IMPLEMENTATION: Direct calculation of D at faces ---
        # # Calculate face temperatures using T_face = ((T_i^4 + T_{i+1}^4) / 2)^(1/4)
        # T_z_face = self._compute_T_face_z(T_n)  # Face temperatures at z-interfaces
        # T_r_face = self._compute_T_face_r(T_n)  # Face temperatures at r-interfaces
        
        # # For z-faces: no material interface (same material on both sides at each r-location)
        # # Calculate D directly at face using face temperature
        # D_z_face = self.D_of_T(T_z_face)
        
        # # For r-faces: detect material interfaces and apply special logic
        # interface_faces_r = self._is_material_interface_r()  # Shape (Nr-1,), True at interfaces
        
        # # Compute D at all r-faces using face temperatures and material interface logic
        # D_r_face = self._compute_D_at_face_with_material_interface(T_r_face, interface_faces_r)
        
        # --- OPTIONAL: Store the old averaging method for backward compatibility ---
        # This is commented out but available if needed for testing/comparison
        if self.kind_of_D_face == "harmonic":
            D_z_face = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + 1e-30)
            D_r_face = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + 1e-30)
        elif self.kind_of_D_face == "arithmetic":
            D_z_face = 0.5 * (D_n[:-1, :] + D_n[1:, :])
            D_r_face = 0.5 * (D_n[:, :-1] + D_n[:, 1:])
        elif self.kind_of_D_face == "geometric":
            D_z_face = np.sqrt(D_n[:-1, :] * D_n[1:, :])
            D_r_face = np.sqrt(D_n[:, :-1] * D_n[:, 1:])
        
        beta_n = self.beta_of_T(T_n)  # β^n
        sigma_n = self.sigma_of_T(T_n)  # σ^n: opacity at time n

        # A^n = β^n Δt χ c σ^n: absorption characteristic
        A_n = beta_n * dt_local * self.chi * self.c * sigma_n
        # C^n = χ c σ^n / (1 + A^n): effective coupling coefficient
        C_n = self.chi * self.c * sigma_n / (1.0 + A_n)

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

        # Radial geometry weight factors (λ^r weights, precomputed)
        lambda_r_mh = self._r_weights["w_mh"]   # λ^r_{j-1/2}
        lambda_r_ph = self._r_weights["w_ph"]  # λ^r_{j+1/2}
        lambda_r_axis = self._r_weights["w_axis"]  # λ^r axis (j=0)
        lambda_r_mh_outer = self._r_weights["w_mh_outer"]  # λ^r_{j-1/2} at outer
        lambda_r_ph_outer = self._r_weights["w_ph_outer"]  # λ^r_{j+1/2} at outer
        dr_outer = outer_dr(self.r)

        inv_Delta_t = 1.0 / dt_local  # 1/Δt
        Delta_z_sq = self.dz * self.dz  # (Δz)^2

        for i in range(i0, i1 + 1):
            base_row = (i - i0) * Nr
            rows = base_row + np.arange(Nr)

            if marshak_boundary and i == 0:
                # Apply Marshak boundary: foam with drive, gold with vacuum
                mask_foam = self.r < self.R_foam
                alpha_vec = 2.0 * D_z_face[0, :] / (self.c * self.dz)
                
                # Foam region: apply Marshak boundary with E_left_drive
                data[pos_self[rows[mask_foam]]] = 1.0 + alpha_vec[mask_foam]
                data[pos_ip[rows[mask_foam]]] = -alpha_vec[mask_foam]
                b[rows[mask_foam]] = self.E_left_drive(t + dt_local)
                
                # Gold region: apply Marshak boundary with vacuum (E_vac = 0)
                data[pos_self[rows[~mask_foam]]] = 1.0 + alpha_vec[~mask_foam]
                data[pos_ip[rows[~mask_foam]]] = -alpha_vec[~mask_foam]
                b[rows[~mask_foam]] = 0.0  # Vacuum boundary: E_vac = 0
                continue

            A_diag = inv_Delta_t + C_n[i, :]  # Diagonal coefficient
            RHS = E_n[i, :] * inv_Delta_t + C_n[i, :] * U_n[i, :]  # RHS vector

            # z-diffusion: ∂E/∂z with finite volume discretization
            D_z_imh = D_z_face[i - 1, :]  # D^n at z-face (i-1/2)
            D_z_iph = D_z_face[i, :]  # D^n at z-face (i+1/2)
            A_diag += (D_z_imh + D_z_iph) / Delta_z_sq

            if i < i1:
                data[pos_ip[rows]] = -D_z_iph / Delta_z_sq
            else:
                RHS += (D_z_iph / Delta_z_sq) * E_right

            if i > i0:
                data[pos_im[rows]] = -D_z_imh / Delta_z_sq
            else:
                if not marshak_boundary:
                    RHS += (D_z_imh / Delta_z_sq) * E_left

            # r-diffusion: cylindrical geometry, 1/r d/dr(r dE/dr)
            if Nr > 1:
                D_r_i = D_r_face[i, :]  # D^n at all r-faces for row i

                # axis j=0: λ^r_{j+1/2} contribution
                lambda_r_axis_n = lambda_r_axis * D_r_i[0]
                A_diag[0] += lambda_r_axis_n
                data[pos_jp[rows[0]]] = -lambda_r_axis_n

                # interior j=1..Nr-2: λ^r_{j-1/2} and λ^r_{j+1/2}
                if Nr > 2:
                    j = np.arange(1, Nr - 1)
                    lambda_r_mh_n = lambda_r_mh[j] * D_r_i[j - 1]  # λ^r_{j-1/2}
                    lambda_r_ph_n = lambda_r_ph[j] * D_r_i[j]  # λ^r_{j+1/2}
                    A_diag[j] += lambda_r_mh_n + lambda_r_ph_n
                    data[pos_jm[rows[j]]] = -lambda_r_mh_n
                    data[pos_jp[rows[j]]] = -lambda_r_ph_n

                # outer j=Nr-1: λ^r_{j-1/2} contribution
                lambda_r_mh_outer_n = lambda_r_mh_outer * D_r_i[-1]
                A_diag[-1] += lambda_r_mh_outer_n
                data[pos_jm[rows[-1]]] = -lambda_r_mh_outer_n
                if bc_r_outer == "dirichlet_bath":
                    lambda_r_ph_outer_n = lambda_r_ph_outer * D_r_i[-1]
                    A_diag[-1] += lambda_r_ph_outer_n
                    RHS[-1] += lambda_r_ph_outer_n * E_right
                elif bc_r_outer == "neumann0":
                    pass
                elif bc_r_outer == "marshak_wall":
                    # Marshak boundary at r=R: (1+α)E_{Nr-1} - α E_{Nr-2} = E_wall
                    alpha_r_marshak = 2.0 * D_r_i[-1] / (self.c * dr_outer + 1e-300)
                    outer_row = rows[-1]
                    if pos_im[outer_row] >= 0:
                        data[pos_im[outer_row]] = 0.0
                    if pos_ip[outer_row] >= 0:
                        data[pos_ip[outer_row]] = 0.0
                    if pos_jp[outer_row] >= 0:
                        data[pos_jp[outer_row]] = 0.0
                    data[pos_jm[outer_row]] = -alpha_r_marshak
                    A_diag[-1] = 1.0 + alpha_r_marshak
                    RHS[-1] = E_right
                else:
                    raise ValueError("bc_r_outer must be 'neumann0', 'dirichlet_bath', or 'marshak_wall'.")

            data[pos_self[rows]] = A_diag
            b[rows] = RHS

        matrix_A = csr_matrix((data, indices, indptr), shape=(n_unknown, n_unknown))

        # Warm start for iterative solver (using E^n as initial guess)
        x0_warm_start = E_n[i0 : i1 + 1, :].reshape(n_unknown)

        # Solve linear system: matrix_A * E_np1_inner = b_rhs
        if self.linear_solver == "direct":
            E_np1_inner = spsolve(matrix_A, b)
        else:
            # Jacobi preconditioner: M = diag(matrix_A)^{-1}
            diag_entries = data[pos_self]
            inv_diag = 1.0 / (diag_entries + 1e-300)

            def precond(v):
                return inv_diag * v

            M = LinearOperator((n_unknown, n_unknown), matvec=precond, dtype=np.float64)
            try:
                # SciPy >= 1.8 uses rtol/atol
                E_np1_inner, info = bicgstab(
                    matrix_A,
                    b,
                    x0=x0_warm_start,
                    rtol=self.linear_tol,
                    atol=0.0,
                    maxiter=self.linear_maxiter,
                    M=M,
                )
            except TypeError:
                # Older SciPy uses tol
                E_np1_inner, info = bicgstab(
                    matrix_A,
                    b,
                    x0=x0_warm_start,
                    tol=self.linear_tol,
                    maxiter=self.linear_maxiter,
                    M=M,
                )
            need_fallback = (info != 0) or (not np.all(np.isfinite(E_np1_inner)))
            if (not need_fallback) and self.linear_check_residual:
                # Check residual: ||A*x - b|| / ||b||
                residual = matrix_A @ E_np1_inner - b
                rel_residual = np.linalg.norm(residual) / (np.linalg.norm(b) + 1e-30)
                if rel_residual > self.linear_residual_factor * self.linear_tol:
                    need_fallback = True

            if need_fallback:
                E_np1_inner = spsolve(matrix_A, b)

        # Reconstruct E^{n+1} on full domain from interior solution
        E_np1 = E_n.copy()
        E_np1[i0 : i1 + 1, :] = E_np1_inner.reshape((nzi, Nr))
        if not marshak_boundary:
            E_np1[0, :] = E_left  # z=0 boundary (Dirichlet drive)
        E_np1[-1, :] = E_right  # z=Lz boundary (Dirichlet bath)

        # Enforce r-axis symmetry explicitly (r=0, dE/dr=0)
        if Nr > 1:
            E_np1[:, 0] = E_np1[:, 1]

        # Update U^{n+1} implicitly (local material energy)
        # U^{n+1} = (A^n * E^{n+1} + U^n) / (1 + A^n)
        U_np1 = (A_n * E_np1 + U_n) / (1.0 + A_n)

        self.E = E_np1
        self.UR = U_np1

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

            Eold = self.E.copy()  # E^n at start of step
            URold = self.UR.copy()  # U^n at start of step

            self.implicit_step(t=t, dt_local=dt_local, bc_r_outer=bc_r_outer, marshak_boundary=marshak_boundary)
            
            # Check for NaN after implicit step
            if not np.all(np.isfinite(self.E)) or not np.all(np.isfinite(self.UR)):
                print(f"\n[FATAL] NaN detected after implicit_step at t={t}")
                print(f"  E finite: {np.all(np.isfinite(self.E))}, UR finite: {np.all(np.isfinite(self.UR))}")
                if not np.all(np.isfinite(self.E)):
                    nan_count = np.sum(~np.isfinite(self.E))
                    print(f"  E has {nan_count} non-finite values")
                if not np.all(np.isfinite(self.UR)):
                    nan_count = np.sum(~np.isfinite(self.UR))
                    print(f"  UR has {nan_count} non-finite values")
                pbar.close()
                raise RuntimeError(f"Simulation diverged at t={t} with NaN values")
            
            t_next = t + dt_local

            U_m = self.U_m_of_UR(self.UR)  # Material internal energy
            T_m = (self.UR / self.a) ** 0.25  # Material temperature
            T_R = (self.E / self.a) ** 0.25  # Radiation temperature
            if store_idx < len(times_to_store) and abs(t_next - times_to_store[store_idx]) < 0.5*dt_local:
                stored_t.append(t_next)
                stored_Um.append(U_m.copy())
                stored_Tm.append(T_m.copy())
                stored_TR.append(T_R.copy())
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
        
        In vacuum mode, the entire radial region is foam (up to R_foam).
        """

        r_nodes = np.asarray(self.r, dtype=float)
        mask_foam = r_nodes < float(self.R_foam)
        if np.count_nonzero(mask_foam) < 2:
            # Fallback: return zeros if mask has < 2 nodes
            Um = np.asarray(stored_Um)
            if Um.ndim == 2:
                return np.array([0.0])
            else:
                return np.zeros(Um.shape[0], dtype=float)
        return self._compute_energy_region(stored_Um, mask_r=mask_foam)

    def compute_energy_gold(self, stored_Um):
        """Total gold energy vs time (axisymmetric), in erg.

        Gold region is defined consistently with the material mask: r >= R_foam.
        Convert to hJ by multiplying by 1e-9 (since 1 erg = 1e-9 hJ).
        
        Returns zero array if no gold region exists (vacuum mode).
        """
        # In vacuum mode (gold_width=0), there's no gold region; return zeros
        if not self.r_info.get('has_gold', True):
            Um = np.asarray(stored_Um)
            if Um.ndim == 2:
                return np.array([0.0])
            else:
                return np.zeros(Um.shape[0], dtype=float)

        r_nodes = np.asarray(self.r, dtype=float)
        mask_gold = r_nodes >= float(self.R_foam)
        if np.count_nonzero(mask_gold) < 2:
            # Fallback: return zeros if mask has < 2 nodes (shouldn't happen with proper grids, but safety)
            Um = np.asarray(stored_Um)
            if Um.ndim == 2:
                return np.array([0.0])
            else:
                return np.zeros(Um.shape[0], dtype=float)
        return self._compute_energy_region(stored_Um, mask_r=mask_gold)

    def compute_heated_gold_cells_by_z(self, stored_Tm, *, threshold: float = 5, T_cold=None):
        """Count how many radial gold cells are heated above a threshold for each z slice.

        Returns
        -------
        counts_by_z : np.ndarray
            Number of gold-region radial cells above threshold for each z index.
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

        if T_cold is None:
            if self.simulation_unit_system == "cgs":
                T_cold = 300.0
            else:
                T_cold = 300.0 / K_per_Hev

        threshold = float(threshold)
        T_cold = float(T_cold)

        mask_gold = np.asarray(self.r, dtype=float) >= float(self.R_foam)
        if np.count_nonzero(mask_gold) == 0:
            return np.zeros(self.Nz, dtype=int)

        T_last = Tm[-1]
        heated = T_last[:, mask_gold] > (threshold * T_cold)
        return np.count_nonzero(heated, axis=1)

    def compute_front_surface(
        self,
        stored_Tm,
        *,
        front_method: str = "maxgrad",
        threshold: float = 10,
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
    if R_foam <= 0:
        raise ValueError("R_foam must be > 0")

    # Allow gold_width <= 0 to represent no gold (vacuum): return foam-only grid
    r_foam = np.linspace(0.0, R_foam, Nr_foam)
    if gold_width is None or gold_width <= 0:
        info = {"has_gold": False, "q": None, "dr0": None, "widths": np.array([]), "R_total": float(R_foam)}
        return r_foam, info

    # From here on we assume a positive gold block
    if Nr_gold < 1:
        raise ValueError("Nr_gold must be >= 1 for a non-zero gold width")
    if dr0 is None:
        raise ValueError("Provide dr0=...")

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

    info = {"has_gold": True, "q": float(q), "dr0": float(dr0), "widths": widths, "R_total": float(R_total)}
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


