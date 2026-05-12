from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parameters import *
from tqdm import tqdm

# find q such that dr0 + dr0*q + dr0*q^2 + ... + dr0*q^(N-1) =~ gold_width
def solve_q_from_dz0(gold_width, N, dz0):
    """
    Solve q >= 1 such that sum_{k=0}^{N-1} dz0*q^k = gold_width.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    if dz0 <= 0:
        raise ValueError("dz0 must be > 0")
    if gold_width <= 0:
        raise ValueError("gold_width must be > 0")
    if dz0 * N > gold_width:
        print(f"dz0*N = {dz0*N} exceeds gold_width = {gold_width}")
        raise ValueError("dz0 too large: even uniform widths N*dz0 exceed gold_width")

    # uniform special case
    if abs(dz0 * N - gold_width) / gold_width < 1e-12:
        return 1.0

    def S(q):
        return dz0 * (q**N - 1.0) / (q - 1.0)

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

class GoldFoam1DSimulation:
    """Stateful 1D gold-foam solver, modeled after the 2D core class."""
    @staticmethod
    def make_z_two_block(Lz, gold_width, Nz_foam, Nr_gold, dz0=None):
        """Build the foam-plus-gold grid used by the 1D solver."""
        if Nz_foam < 2:
            raise ValueError("Nz_foam must be >= 2")
        if Lz <= 0:
            raise ValueError("Lz must be > 0")

        z_foam = np.linspace(0.0, Lz, Nz_foam)
        if gold_width is None or gold_width <= 0:
            info = {"has_gold": False, "q": None, "dz0": None, "widths": np.array([]), "L_total": float(Lz)}
            return z_foam, info

        if Nr_gold < 1:
            raise ValueError("Nr_gold must be >= 1 for a non-zero gold width")
        if dz0 is None:
            raise ValueError("Provide dz0=...")

        q = solve_q_from_dz0(gold_width, Nr_gold, dz0)
        widths = dz0 * (q ** np.arange(Nr_gold))
        z_gold = Lz + np.concatenate(([0.0], np.cumsum(widths)))

        L_total = Lz + gold_width
        z_gold[-1] = L_total
        z = np.concatenate((z_foam, z_gold))
        z = np.unique(z)

        info = {"has_gold": True, "q": float(q), "dz0": float(dz0), "widths": widths, "L_total": float(L_total)}
        return z, info

    def __init__(
        self,
        *,
        nz: int | None = None,
        lz: float | None = None,
        gold_block_width: float | None = None,
        dt_init: float | None = None,
        t_final_override: float | None = None,
        simulation_unit_system_override: str | None = None,
        kind_of_D_face_override: str | None = None,
        chi_override: float | None = None,
    ):
        self.simulation_unit_system = simulation_unit_system if simulation_unit_system_override is None else simulation_unit_system_override
        self.kind_of_D_face = kind_of_D_face if kind_of_D_face_override is None else kind_of_D_face_override

        self.Lz = float(L if lz is None else lz)
        self.gold_width = float(w_Au if gold_block_width is None else gold_block_width)
        self.Nz_foam = int(500 if nz is None else nz)
        if self.Nz_foam < 2:
            raise ValueError("nz must be >= 2")

        self.Nr_gold = 30
        dz0 = self.gold_width / 3000 if self.gold_width > 0 else None
        self.z, self.z_info = self.make_z_two_block(self.Lz, self.gold_width, self.Nz_foam, self.Nr_gold, dz0=dz0)
        self.L_total = float(self.z_info["L_total"])
        self.Nz = self.z.size
        self.z_foam: np.ndarray = np.linspace(0.0, self.Lz, self.Nz_foam)
        self.dz = float(self.z_foam[1] - self.z_foam[0])
        self.mask_foam = self.z < self.Lz

        self.f_profile = np.where(self.mask_foam, f, f_gold)
        self.g_profile = np.where(self.mask_foam, g, g_gold)
        self.alpha_profile = np.where(self.mask_foam, alpha, alpha_gold)
        self.beta_profile = np.where(self.mask_foam, beta, beta_gold)
        self.lambda_profile = np.where(self.mask_foam, lambda_param, lambda_param_gold)
        self.mu_profile = np.where(self.mask_foam, mu, mu_gold)
        self.rho_profile = np.where(self.mask_foam, rho, rho_gold)

        self.chi = float(chi if chi_override is None else chi_override)
        self.dt = float(dt if dt_init is None else dt_init)
        self.t_final = float(t_final if t_final_override is None else t_final_override)

        if self.simulation_unit_system == CGS:
            self.a = a_kelvin
            self.c = 3e10
            self.T_material_0 = float(T_material_0_Kelvin)
        elif self.simulation_unit_system == HEV_NS:
            self.a = a_hev
            self.c = 30.0
            self.T_material_0 = float(T_material_0_hev)
        else:
            raise ValueError("simulation_unit_system must be CGS or HEV_NS")

        self.data_dir = os.path.join(os.path.dirname(__file__), "Data_new", "Experiment", "Material", "1D_simulation")
        self.E, self.UR = self.init_state()

    def init_state(self):
        E0 = self.a * self.T_material_0**4 * np.ones(self.Nz)
        UR0 = self.a * self.T_material_0**4 * np.ones(self.Nz)
        return E0, UR0

    def sigma_of_T(self, T):
        if self.simulation_unit_system == CGS:
            T_Hev = T / K_per_Hev
            return 1.0 / (self.g_profile * T_Hev ** self.alpha_profile * self.rho_profile ** (-self.lambda_profile - 1))
        return 1.0 / (self.g_profile * T ** self.alpha_profile * self.rho_profile ** (-self.lambda_profile - 1))

    def beta_of_T(self, T):
        if self.simulation_unit_system == CGS:
            Cv_m = self.f_profile * self.beta_profile * T ** (self.beta_profile - 1) * self.rho_profile ** (-self.mu_profile + 1)
            Cv_R = 4.0 * self.a * T ** 3
            return Cv_R / Cv_m * K_per_Hev ** self.beta_profile
        return ((4.0 * self.a * self.rho_profile ** (self.mu_profile - 1)) / (self.f_profile * self.beta_profile)) * T ** (4.0 - self.beta_profile)

    def D_of_T(self, T):
        return self.c / (3.0 * self.sigma_of_T(T))

    def U_m_of_T(self, UR):
        if self.simulation_unit_system == CGS:
            T_Hev = (UR / self.a) ** 0.25 / K_per_Hev
            return self.f_profile * T_Hev ** self.beta_profile * self.rho_profile ** (-self.mu_profile + 1)
        T_Hev = (UR / self.a) ** 0.25
        return self.f_profile * T_Hev ** self.beta_profile * self.rho_profile ** (-self.mu_profile + 1)

    def _boundary_energy(self, t):
        if self.simulation_unit_system == CGS:
            t_ns = t * 1e9
            left_T = K_per_Hev * get_TD(t_ns, t_array_TD, T_array_TD)
            right_T = 300.0
        else:
            left_T = get_TD(t, t_array_TD, T_array_TD)
            right_T = 300.0 / K_per_Hev
        return self.a * left_T**4, self.a * right_T**4

    def implicit_step(self, E, UR, *, t=0.0, dt_local=None, marshak_boundary=False):
        if dt_local is None:
            dt_local = self.dt

        N = E.size
        n_int = N - 1 if marshak_boundary else N - 2
        E_left, E_right = self._boundary_energy(t)
        Tn = (UR / self.a) ** 0.25
        Dn = self.D_of_T(Tn)
        betan = self.beta_of_T(Tn)
        sigman = self.sigma_of_T(Tn)

        if self.kind_of_D_face == "harmonic":
            D_face = 2.0 * Dn[:-1] * Dn[1:] / (Dn[:-1] + Dn[1:] + 1e-20)
        elif self.kind_of_D_face == "arithmetic":
            D_face = (Dn[:-1] + Dn[1:]) / 2.0
        elif self.kind_of_D_face == "geometric":
            D_face = np.sqrt(Dn[:-1] * Dn[1:])
        else:
            raise ValueError("kind_of_D_face must be harmonic, arithmetic, or geometric")

        A = betan * dt_local * self.chi * self.c * sigman
        coupling = self.chi * self.c * sigman / (1.0 + A)

        lower = np.zeros(max(n_int - 1, 0))
        diag = np.zeros(n_int)
        upper = np.zeros(max(n_int - 1, 0))
        rhs = np.zeros(n_int)

        if marshak_boundary:
            diag[0] = 1.0 + 2.0 * D_face[0] / (self.c * self.dz)
            if upper.size:
                upper[0] = -2.0 * D_face[0] / (self.c * self.dz)
            rhs[0] = self.a * (E_left / self.a)

            for k in range(1, n_int):
                i = k
                D_imh = D_face[i - 1]
                D_iph = D_face[i]
                a_i = -D_imh / self.dz**2
                c_i = -D_iph / self.dz**2
                b_i = (1.0 / dt_local) + (D_imh + D_iph) / self.dz**2 + coupling[i]
                d_i = (E[i] / dt_local) + coupling[i] * UR[i]
                diag[k] = b_i
                rhs[k] = d_i
                lower[k - 1] = a_i
                if k < n_int - 1:
                    upper[k] = c_i
        else:
            for k in range(n_int):
                i = k + 1
                D_imh = D_face[i - 1]
                D_iph = D_face[i]
                a_i = -D_imh / self.dz**2
                c_i = -D_iph / self.dz**2
                b_i = (1.0 / dt_local) + (D_imh + D_iph) / self.dz**2 + coupling[i]
                d_i = (E[i] / dt_local) + coupling[i] * UR[i]
                diag[k] = b_i
                rhs[k] = d_i
                if k > 0:
                    lower[k - 1] = a_i
                if k < n_int - 1:
                    upper[k] = c_i

            rhs[0] -= (-D_face[0] / self.dz**2) * E_left

        rhs[-1] -= (-D_face[-1] / self.dz**2) * E_right

        for i in range(1, n_int):
            w = lower[i - 1] / diag[i - 1]
            diag[i] -= w * upper[i - 1]
            rhs[i] -= w * rhs[i - 1]

        E_inner = np.empty(n_int)
        E_inner[-1] = rhs[-1] / diag[-1]
        for i in range(n_int - 2, -1, -1):
            E_inner[i] = (rhs[i] - upper[i] * E_inner[i + 1]) / diag[i]

        E_new = E.copy()
        E_new[-1] = E_right
        if marshak_boundary:
            E_new[:-1] = E_inner
        else:
            E_new[0] = E_left
            E_new[1:-1] = E_inner

        UR_new = (A * E_new + UR) / (1.0 + A)
        return E_new, UR_new

    def run(self, times_to_store, *, dtfac=0.05, dtmin=5e-15, dtmax=2e-13, marshak_boundary=False):
        store_idx = 0
        stored_t, stored_Um, stored_Tm, stored_TR = [], [], [], []
        t = 0.0
        dt_local = self.dt
        pbar = tqdm(total=self.t_final, desc="Simulating", unit="s", ncols=100)

        while t < self.t_final - 1e-30:
            dt_local = min(dt_local, self.t_final - t)
            if store_idx < len(times_to_store):
                t_target = times_to_store[store_idx]
                if t < t_target <= t + dt_local:
                    dt_local = t_target - t

            Eold = self.E.copy()
            URold = self.UR.copy()
            self.E, self.UR = self.implicit_step(self.E, self.UR, t=t, dt_local=dt_local, marshak_boundary=marshak_boundary)
            t_next = t + dt_local

            Um = self.U_m_of_T(self.UR)
            Tm = (self.UR / self.a) ** 0.25
            TR = (self.E / self.a) ** 0.25

            if store_idx < len(times_to_store) and abs(t_next - times_to_store[store_idx]) < 0.5 * dt_local:
                if self.simulation_unit_system == CGS:
                    stored_Um.append(np.array(Um).copy())
                    stored_Tm.append(np.array((Tm / K_per_Hev)).copy())
                    stored_t.append(t_next * 1e9)
                    stored_TR.append(np.array((TR / K_per_Hev)).copy())
                else:
                    stored_Um.append(np.array(Um).copy())
                    stored_Tm.append(np.array(Tm).copy())
                    stored_TR.append(np.array(TR).copy())
                    stored_t.append(t_next)
                store_idx += 1

            dt_new, _, _ = update_dt_relchange(dt_local, self.E, Eold, self.UR, URold, dtfac=dtfac, dtmax=dtmax)
            if dtmin is not None:
                dt_new = max(dt_new, dtmin)
            pbar.update(t_next - t)
            t = t_next
            dt_local = dt_new

        pbar.close()
        return np.array(stored_t), np.array(stored_Um), np.array(stored_Tm), np.array(stored_TR)

    def compute_front_and_energy(self, stored_Um, stored_Tm):
        front_positions = []
        total_energies = []
        for Ti, Ui in zip(stored_Tm, stored_Um):
            front_idx = np.argmax(np.abs(np.diff(Ti)))
            front_positions.append(self.z[front_idx])
            total_energy = np.trapezoid(Ui, self.z)
            # hJ = 10^2 J
            # erg = 10^-7 J = 10^-9 hJ
            # 1 / cm^2 = 10^-2 / mm^2
            # => erg/cm^2 = 10^-11 hJ/mm^2
            # => integrate Um (erg/cm^3) over z (cm) gives erg/cm^2 = 10^-11 hJ/mm^2
            total_energies.append(total_energy * 1e-11)
        return np.array(front_positions), np.array(total_energies)

    def save_outputs(self, stored_t, stored_Um, stored_Tm, stored_TR, *, marshak_boundary=True):
        os.makedirs(self.data_dir, exist_ok=True)
        suffix = "marshak" if marshak_boundary else "linear"
        pd.DataFrame(stored_Tm).to_csv(os.path.join(self.data_dir, f"stored_Tm_{suffix}.csv"), header=False, index=False)
        pd.DataFrame(stored_TR).to_csv(os.path.join(self.data_dir, f"stored_TR_{suffix}.csv"), header=False, index=False)
        pd.DataFrame(stored_Um).to_csv(os.path.join(self.data_dir, f"stored_Um_{suffix}.csv"), header=False, index=False)
        pd.DataFrame(stored_t).to_csv(os.path.join(self.data_dir, f"stored_time_{suffix}.csv"), header=False, index=False)
