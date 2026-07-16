"""
Solvers module for the Unified Gray Hammer-Rosen (HR) Model.

Implements the complete 7-step recipe from Section 10 of:
'A Unified Gray Hammer-Rosen Model for Surface- and Bath-Temperature Drives'.

Provides:
- `SurfaceDriveSolver`: Solves for prescribed surface radiation temperature T_s(t).
- `BathDriveSolver`: Solves for prescribed bath temperature T_bath(t) plus coupled T_s(t).
- `GrayHRSolution`: Rich solution container with algebraic profile evaluators.
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from typing import Union, Callable, Dict, Any, Optional, Tuple

from parameters import GrayHRParameters
from moments import compute_G, compute_M_p, compute_N_p


class GrayHRSolution:
    """
    Rich solution container for the Unified Gray HR Model.
    """
    def __init__(
        self,
        t: np.ndarray,
        x_F0: np.ndarray,
        x_F: np.ndarray,
        T_m: np.ndarray,
        T_s: np.ndarray,
        B: np.ndarray,
        s: np.ndarray,
        U_m: np.ndarray,
        params: GrayHRParameters,
        drive_type: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        self.t = t
        self.x_F0 = x_F0
        self.x_F = x_F
        self.T_m = T_m
        self.T_s = T_s
        self.B = B
        self.s = s
        self.U_m = U_m
        self.params = params
        self.drive_type = drive_type
        self.extra = extra or {}

    def get_radiation_profile(self, time_or_idx: Union[float, int], x: np.ndarray) -> np.ndarray:
        """
        Evaluates the spatial radiation temperature profile T_r(x, t) at a given time or index.
        
        Formula (Step 7):
            T_r(x, t) = T_s(t) * [ (1 - x/x_F) * (1 + B * x/x_F) ]^(1 / (4 + alpha - beta))
            for 0 <= x <= x_F(t), otherwise 0.
        """
        idx = self._resolve_index(time_or_idx)
        xf_val = self.x_F[idx]
        ts_val = self.T_s[idx]
        b_val = self.B[idx]
        exp_val = self.params.front_exponent
        
        if xf_val <= 0.0:
            return np.zeros_like(x, dtype=float)
            
        y = np.clip(x / xf_val, 0.0, 1.0)
        base = (1.0 - y) * (1.0 + b_val * y)
        base = np.maximum(0.0, base)
        
        tr = ts_val * (base ** exp_val)
        tr[x > xf_val] = 0.0
        return tr

    def get_material_profile(self, time_or_idx: Union[float, int], x: np.ndarray) -> np.ndarray:
        """
        Evaluates the spatial material temperature profile T(x, t) at a given time or index.
        
        Formula (Step 7):
            T(x, t) = T_m(t) * [ (1 - x/x_F) * (1 + B * x/x_F) ]^(1 / (4 + alpha - beta)) * exp(-s * x/x_F)
            for 0 <= x <= x_F(t), otherwise 0.
        """
        idx = self._resolve_index(time_or_idx)
        xf_val = self.x_F[idx]
        tm_val = self.T_m[idx]
        b_val = self.B[idx]
        s_val = self.s[idx]
        exp_val = self.params.front_exponent
        
        if xf_val <= 0.0:
            return np.zeros_like(x, dtype=float)
            
        y = np.clip(x / xf_val, 0.0, 1.0)
        base = (1.0 - y) * (1.0 + b_val * y)
        base = np.maximum(0.0, base)
        
        tm = tm_val * (base ** exp_val) * np.exp(-s_val * y)
        tm[x > xf_val] = 0.0
        return tm

    def _resolve_index(self, time_or_idx: Union[float, int]) -> int:
        if isinstance(time_or_idx, (int, np.integer)):
            return int(time_or_idx)
        return int(np.argmin(np.abs(self.t - time_or_idx)))


def solve_material_lag(
    B: float,
    target: float,
    beta: float,
    front_exponent: float,
    last_s: float = 0.0
) -> float:
    """
    Solves M_beta(B, beta * s) = target for s (Step 5).
    Unique root since M_beta is strictly decreasing in s.
    """
    if target <= 0.0:
        return 0.0
        
    val_0 = compute_M_p(B, 0.0, beta, front_exponent)
    if abs(val_0 - target) < 1e-12 * max(1.0, target):
        return 0.0

    def residual(s_val):
        return compute_M_p(B, beta * s_val, beta, front_exponent) - target

    s_low, s_high = 0.0, 0.0
    if val_0 > target:
        s_low = 0.0
        s_high = max(2.0, 2.0 * abs(last_s))
        while residual(s_high) > 0.0:
            s_low = s_high
            s_high *= 2.0
            if s_high > 100.0:
                s_high = 100.0
                break
    else:
        s_high = 0.0
        s_low = -max(2.0, 2.0 * abs(last_s))
        while residual(s_low) < 0.0:
            s_high = s_low
            s_low *= 2.0
            if s_low < -20.0:
                s_low = -20.0
                break

    try:
        r_low, r_high = residual(s_low), residual(s_high)
        if (r_low <= 0.0 <= r_high) or (r_high <= 0.0 <= r_low):
            from scipy.optimize import brentq
            return float(brentq(residual, s_low, s_high, xtol=1e-5))
        else:
            return s_high if abs(r_high) < abs(r_low) else s_low
    except Exception:
        return float(last_s)


def _get_effective_c_and_C(params: GrayHRParameters, use_seconds: bool) -> Tuple[float, float]:
    """
    Returns effective speed of light (c_eff) and leading coefficient C_eff
    consistent with time units (seconds vs nanoseconds).
    """
    c_eff = params.c if use_seconds else (params.c * 1e-9)
    C_eff = (4.0 * params.a_rad * c_eff * params.g) / (
        3.0 * (4.0 + params.alpha) * params.f
    ) * (params.rho ** (-2.0 + params.mu - params.lambda_param))
    return c_eff, C_eff


class SurfaceDriveSolver:
    """
    Solver for prescribed surface radiation temperature T_s(t).
    """
    def __init__(self, params: GrayHRParameters):
        self.params = params

    def solve(
        self,
        times: np.ndarray,
        T_s_drive: Union[np.ndarray, Callable[[float], float], tuple],
        T_init: Optional[float] = None,
        method: str = "Radau",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        use_seconds: bool = False
    ) -> GrayHRSolution:
        t_span = (float(times[0]), float(times[-1]))
        c_eff, C_eff = _get_effective_c_and_C(self.params, use_seconds)
        
        if callable(T_s_drive):
            def T_s_func(t): return max(1.0, float(T_s_drive(t)))
            def dT_s_dt_func(t):
                dt = 1e-5 * max(1.0, abs(t))
                return (T_s_func(t + dt) - T_s_func(t - dt)) / (2.0 * dt)
        elif isinstance(T_s_drive, tuple):
            t_data, T_data = T_s_drive
            interp = PchipInterpolator(t_data, np.maximum(1.0, T_data))
            dT_s_interp = interp.derivative()
            def T_s_func(t): return max(1.0, float(interp(t)))
            def dT_s_dt_func(t): return float(dT_s_interp(t))
        else:
            interp = PchipInterpolator(times, np.maximum(1.0, T_s_drive))
            dT_s_interp = interp.derivative()
            def T_s_func(t): return max(1.0, float(interp(t)))
            def dT_s_dt_func(t): return float(dT_s_interp(t))

        ts_0 = max(1.0, T_s_func(t_span[0]))
        tm_0 = ts_0 if T_init is None else max(1.0, float(T_init))
        
        # Coupled state vector: y = [T_m, U_m, I]
        y0 = np.array([tm_0, 0.0, 0.0], dtype=float)
        last_s = [0.0]

        def rhs(t, y):
            tm = max(1.0, y[0])
            um = max(0.0, y[1])
            i_quad = max(0.0, y[2])
            
            ts = max(1.0, T_s_func(t))
            dts_dt = dT_s_dt_func(t)
            tm_reg = max(tm, 0.2 * ts, 1.0)
            
            p = self.params
            R = (p.a_rad * (ts ** 4)) / (p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta))
            G_R = compute_G(R, p.alpha, p.beta)
            
            dI_dt = (ts ** 4) * (tm ** p.alpha) * G_R
            xf0 = np.sqrt(max(0.0, ((2.0 + p.eps) / (1.0 - p.eps)) * C_eff * (tm_reg ** -p.beta) * i_quad) + 1e-16)
            
            if i_quad <= 1e-20 or ts <= 1e-8 or xf0 <= 1e-15:
                B = p.eps / 2.0
            else:
                d_ln_ts_4alpha = (4.0 + p.alpha) * dts_dt / ts
                B = (p.eps / 2.0) * (
                    1.0 - ((2.0 + p.eps) / (1.0 - p.eps)) * 
                    (i_quad / ((ts ** 4) * (tm ** p.alpha) * G_R)) * d_ln_ts_4alpha
                )
            B = max(-0.99999, B)
            
            denom = max(1e-12, xf0 * p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta))
            target = um / denom
            s_val = solve_material_lag(B, target, p.beta, p.front_exponent, last_s=last_s[0])
            last_s[0] = s_val
            
            dTm_dt = (p.a_rad * c_eff) / (p.beta * p.f * p.g_prime) * (
                p.rho ** (p.mu + p.lambda_prime)
            ) * (tm_reg ** (1.0 - p.alpha_prime - p.beta)) * ((ts ** 4) - (tm ** 4))
            
            m1 = compute_M_p(B, -p.alpha_prime * s_val, 4.0 - p.alpha_prime, p.front_exponent)
            m2 = compute_M_p(B, (4.0 - p.alpha_prime) * s_val, 4.0 - p.alpha_prime, p.front_exponent)
            dUm_dt = (p.a_rad * c_eff / p.g_prime) * (p.rho ** (1.0 + p.lambda_prime)) * xf0 * (
                (ts ** 4) * (tm_reg ** -p.alpha_prime) * m1 - (tm ** (4.0 - p.alpha_prime)) * m2
            )
            
            return [dTm_dt, dUm_dt, dI_dt]

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            t_eval=times,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        if not sol.success:
            t_last = sol.t[-1] if len(sol.t) > 0 else t_span[0]
            y_last = sol.y[:, -1] if sol.y.shape[1] > 0 else y0
            try:
                dy_last = rhs(t_last, y_last)
            except Exception as e:
                dy_last = f"error in rhs: {e}"
            raise RuntimeError(f"ODE integration failed at t={t_last}:\n  Message: {sol.message}\n  State y={y_last}\n  RHS dy={dy_last}")
            
        return self._post_process(sol.t, sol.y, T_s_func, c_eff, C_eff)

    def _post_process(
        self,
        t_arr: np.ndarray,
        y_arr: np.ndarray,
        T_s_func: Callable[[float], float],
        c_eff: float,
        C_eff: float
    ) -> GrayHRSolution:
        p = self.params
        N = len(t_arr)
        
        tm_arr = np.maximum(1.0, y_arr[0])
        um_arr = np.maximum(0.0, y_arr[1])
        i_arr = np.maximum(0.0, y_arr[2])
        ts_arr = np.maximum(1.0, np.array([T_s_func(ti) for ti in t_arr], dtype=float))
        
        xf0_arr = np.zeros(N, dtype=float)
        xf_arr = np.zeros(N, dtype=float)
        b_arr = np.zeros(N, dtype=float)
        s_arr = np.zeros(N, dtype=float)
        ds_arr = np.zeros(N, dtype=float)
        d0_arr = np.zeros(N, dtype=float)
        
        last_s = 0.0
        for k in range(N):
            ti = t_arr[k]
            tm = tm_arr[k]
            ts = ts_arr[k]
            i_q = i_arr[k]
            tm_reg = max(tm, 0.2 * ts, 1.0)
            
            xf0 = np.sqrt(max(0.0, ((2.0 + p.eps) / (1.0 - p.eps)) * C_eff * (tm_reg ** -p.beta) * i_q))
            xf0_arr[k] = xf0
            
            R = (p.a_rad * (ts ** 4)) / (p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta))
            G_R = compute_G(R, p.alpha, p.beta)
            
            dt = 1e-5 * max(1.0, abs(ti))
            dts_dt = (T_s_func(ti + dt) - T_s_func(ti - dt)) / (2.0 * dt)
            
            if i_q <= 1e-20 or ts <= 1e-8 or xf0 <= 1e-15:
                b_val = p.eps / 2.0
            else:
                d_ln_ts_4alpha = (4.0 + p.alpha) * dts_dt / ts
                b_val = (p.eps / 2.0) * (
                    1.0 - ((2.0 + p.eps) / (1.0 - p.eps)) * 
                    (i_q / ((ts ** 4) * (tm ** p.alpha) * G_R)) * d_ln_ts_4alpha
                )
            b_val = max(-0.99999, b_val)
            b_arr[k] = b_val
            
            denom = max(1e-12, xf0 * p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta))
            target = um_arr[k] / denom
            s_val = solve_material_lag(b_val, target, p.beta, p.front_exponent, last_s=last_s)
            s_arr[k] = s_val
            last_s = s_val
            
            mb_s = compute_M_p(b_val, p.alpha * s_val, p.beta, p.front_exponent)
            nb_s = compute_N_p(b_val, p.alpha * s_val, p.beta, p.front_exponent)
            ds_arr[k] = (ts ** 4) * (tm ** p.alpha) * ((1.0 - b_val) * mb_s + 2.0 * b_val * nb_s)
            
            mb_0 = compute_M_p(b_val, 0.0, p.beta, p.front_exponent)
            nb_0 = compute_N_p(b_val, 0.0, p.beta, p.front_exponent)
            d0_arr[k] = (ts ** 4) * (tm ** p.alpha) * ((1.0 - b_val) * mb_0 + 2.0 * b_val * nb_0)
            
        inum_arr = cumulative_trapezoid(d0_arr, t_arr, initial=0.0)
        iden_arr = cumulative_trapezoid(ds_arr, t_arr, initial=0.0)
        
        for k in range(N):
            xf0 = xf0_arr[k]
            if xf0 <= 1e-15:
                xf_arr[k] = 0.0
            else:
                b_val = b_arr[k]
                tm = tm_arr[k]
                ts = ts_arr[k]
                s_val = s_arr[k]
                
                n4_0 = compute_N_p(b_val, 0.0, 4.0, p.front_exponent)
                nb_0 = compute_N_p(b_val, 0.0, p.beta, p.front_exponent)
                nb_s = compute_N_p(b_val, p.beta * s_val, p.beta, p.front_exponent)
                
                e1_0 = p.a_rad * (ts ** 4) * n4_0 + p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta) * nb_0
                e1_s = p.a_rad * (ts ** 4) * n4_0 + p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta) * nb_s
                
                ratio_e1 = e1_0 / max(1e-30, e1_s)
                ratio_i = inum_arr[k] / max(1e-30, iden_arr[k]) if iden_arr[k] > 0 else 1.0
                xf_arr[k] = xf0 * np.sqrt(max(0.0, ratio_e1 * ratio_i))

        return GrayHRSolution(
            t=t_arr,
            x_F0=xf0_arr,
            x_F=xf_arr,
            T_m=tm_arr,
            T_s=ts_arr,
            B=b_arr,
            s=s_arr,
            U_m=um_arr,
            params=p,
            drive_type="surface"
        )


class BathDriveSolver:
    """
    Solver for prescribed bath temperature T_bath(t).
    """
    def __init__(self, params: GrayHRParameters):
        self.params = params

    def solve(
        self,
        times: np.ndarray,
        T_bath_drive: Union[np.ndarray, Callable[[float], float], tuple],
        T_init: Optional[float] = None,
        method: str = "Radau",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        use_seconds: bool = False
    ) -> GrayHRSolution:
        t_span = (float(times[0]), float(times[-1]))
        c_eff, C_eff = _get_effective_c_and_C(self.params, use_seconds)
        
        if callable(T_bath_drive):
            def T_bath_func(t): return max(1.0, float(T_bath_drive(t)))
        elif isinstance(T_bath_drive, tuple):
            t_data, T_data = T_bath_drive
            interp = PchipInterpolator(t_data, np.maximum(1.0, T_data))
            def T_bath_func(t): return max(1.0, float(interp(t)))
        else:
            interp = PchipInterpolator(times, np.maximum(1.0, T_bath_drive))
            def T_bath_func(t): return max(1.0, float(interp(t)))

        tb_0 = max(1.0, T_bath_func(t_span[0]))
        ts_0 = tb_0 if T_init is None else max(1.0, float(T_init))
        tm_0 = ts_0
        
        # Coupled state vector: y = [T_m, T_s, U_m, I]
        y0 = np.array([tm_0, ts_0, 0.0, 0.0], dtype=float)
        last_s = [0.0]

        def rhs(t, y):
            tm = max(1.0, y[0])
            ts = max(1.0, y[1])
            um = max(0.0, y[2])
            i_quad = max(0.0, y[3])
            
            tb = max(1.0, T_bath_func(t))
            tm_reg = max(tm, 0.2 * ts, 1.0)
            
            p = self.params
            R = (p.a_rad * (ts ** 4)) / (p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta))
            G_R = compute_G(R, p.alpha, p.beta)
            
            dI_dt = (ts ** 4) * (tm ** p.alpha) * G_R
            xf0 = np.sqrt(max(0.0, ((2.0 + p.eps) / (1.0 - p.eps)) * C_eff * (tm_reg ** -p.beta) * i_quad) + 1e-16)
            
            if xf0 <= 1e-15 or ts <= 1e-10:
                B = 1.0
            else:
                factor = (3.0 * (4.0 + p.alpha - p.beta) * xf0 * (p.rho ** (1.0 + p.lambda_param))) / (
                    8.0 * p.g * (tm ** p.alpha)
                )
                B = 1.0 - factor * (((tb ** 4) / (ts ** 4)) - 1.0)
            B = max(-0.99999, B)
            
            if i_quad <= 1e-20 or ts <= 1e-8:
                dTs_dt = 0.0
            else:
                coeff = (2.0 * (ts ** 5) * (tm ** p.alpha) * G_R) / (
                    p.eps * (4.0 + p.alpha) * ((2.0 + p.eps) / (1.0 - p.eps)) * i_quad + 1e-30
                )
                dTs_dt = coeff * (B - 1.0 + p.eps / 2.0)
                
            denom = max(1e-12, xf0 * p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta))
            target = um / denom
            s_val = solve_material_lag(B, target, p.beta, p.front_exponent, last_s=last_s[0])
            last_s[0] = s_val
            
            dTm_dt = (p.a_rad * c_eff) / (p.beta * p.f * p.g_prime) * (
                p.rho ** (p.mu + p.lambda_prime)
            ) * (tm_reg ** (1.0 - p.alpha_prime - p.beta)) * ((ts ** 4) - (tm ** 4))
            
            m1 = compute_M_p(B, -p.alpha_prime * s_val, 4.0 - p.alpha_prime, p.front_exponent)
            m2 = compute_M_p(B, (4.0 - p.alpha_prime) * s_val, 4.0 - p.alpha_prime, p.front_exponent)
            dUm_dt = (p.a_rad * c_eff / p.g_prime) * (p.rho ** (1.0 + p.lambda_prime)) * xf0 * (
                (ts ** 4) * (tm_reg ** -p.alpha_prime) * m1 - (tm ** (4.0 - p.alpha_prime)) * m2
            )
            
            return [dTm_dt, dTs_dt, dUm_dt, dI_dt]

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            t_eval=times,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        if not sol.success:
            t_last = sol.t[-1] if len(sol.t) > 0 else t_span[0]
            y_last = sol.y[:, -1] if sol.y.shape[1] > 0 else y0
            try:
                dy_last = rhs(t_last, y_last)
            except Exception as e:
                dy_last = f"error in rhs: {e}"
            raise RuntimeError(f"ODE integration failed at t={t_last}:\n  Message: {sol.message}\n  State y={y_last}\n  RHS dy={dy_last}")
            
        return self._post_process(sol.t, sol.y, T_bath_func, c_eff, C_eff)

    def _post_process(
        self,
        t_arr: np.ndarray,
        y_arr: np.ndarray,
        T_bath_func: Callable[[float], float],
        c_eff: float,
        C_eff: float
    ) -> GrayHRSolution:
        p = self.params
        N = len(t_arr)
        
        tm_arr = np.maximum(1.0, y_arr[0])
        ts_arr = np.maximum(1.0, y_arr[1])
        um_arr = np.maximum(0.0, y_arr[2])
        i_arr = np.maximum(0.0, y_arr[3])
        
        xf0_arr = np.zeros(N, dtype=float)
        xf_arr = np.zeros(N, dtype=float)
        b_arr = np.zeros(N, dtype=float)
        s_arr = np.zeros(N, dtype=float)
        ds_arr = np.zeros(N, dtype=float)
        d0_arr = np.zeros(N, dtype=float)
        
        last_s = 0.0
        for k in range(N):
            ti = t_arr[k]
            tm = tm_arr[k]
            ts = ts_arr[k]
            tb = T_bath_func(ti)
            i_q = i_arr[k]
            tm_reg = max(tm, 0.2 * ts, 1.0)
            
            xf0 = np.sqrt(max(0.0, ((2.0 + p.eps) / (1.0 - p.eps)) * C_eff * (tm_reg ** -p.beta) * i_q))
            xf0_arr[k] = xf0
            
            if xf0 <= 1e-15 or ts <= 1e-10:
                b_val = 1.0
            else:
                factor = (3.0 * (4.0 + p.alpha - p.beta) * xf0 * (p.rho ** (1.0 + p.lambda_param))) / (
                    8.0 * p.g * (tm ** p.alpha)
                )
                b_val = 1.0 - factor * (((tb ** 4) / (ts ** 4)) - 1.0)
            b_val = max(-0.99999, b_val)
            b_arr[k] = b_val
            
            denom = max(1e-12, xf0 * p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta))
            target = um_arr[k] / denom
            s_val = solve_material_lag(b_val, target, p.beta, p.front_exponent, last_s=last_s)
            s_arr[k] = s_val
            last_s = s_val
            
            mb_s = compute_M_p(b_val, p.alpha * s_val, p.beta, p.front_exponent)
            nb_s = compute_N_p(b_val, p.alpha * s_val, p.beta, p.front_exponent)
            ds_arr[k] = (ts ** 4) * (tm ** p.alpha) * ((1.0 - b_val) * mb_s + 2.0 * b_val * nb_s)
            
            mb_0 = compute_M_p(b_val, 0.0, p.beta, p.front_exponent)
            nb_0 = compute_N_p(b_val, 0.0, p.beta, p.front_exponent)
            d0_arr[k] = (ts ** 4) * (tm ** p.alpha) * ((1.0 - b_val) * mb_0 + 2.0 * b_val * nb_0)
            
        inum_arr = cumulative_trapezoid(d0_arr, t_arr, initial=0.0)
        iden_arr = cumulative_trapezoid(ds_arr, t_arr, initial=0.0)
        
        for k in range(N):
            xf0 = xf0_arr[k]
            if xf0 <= 1e-15:
                xf_arr[k] = 0.0
            else:
                b_val = b_arr[k]
                tm = tm_arr[k]
                ts = ts_arr[k]
                s_val = s_arr[k]
                
                n4_0 = compute_N_p(b_val, 0.0, 4.0, p.front_exponent)
                nb_0 = compute_N_p(b_val, 0.0, p.beta, p.front_exponent)
                nb_s = compute_N_p(b_val, p.beta * s_val, p.beta, p.front_exponent)
                
                e1_0 = p.a_rad * (ts ** 4) * n4_0 + p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta) * nb_0
                e1_s = p.a_rad * (ts ** 4) * n4_0 + p.f * (p.rho ** (1.0 - p.mu)) * (tm ** p.beta) * nb_s
                
                ratio_e1 = e1_0 / max(1e-30, e1_s)
                ratio_i = inum_arr[k] / max(1e-30, iden_arr[k]) if iden_arr[k] > 0 else 1.0
                xf_arr[k] = xf0 * np.sqrt(max(0.0, ratio_e1 * ratio_i))

        return GrayHRSolution(
            t=t_arr,
            x_F0=xf0_arr,
            x_F=xf_arr,
            T_m=tm_arr,
            T_s=ts_arr,
            B=b_arr,
            s=s_arr,
            U_m=um_arr,
            params=p,
            drive_type="bath"
        )
