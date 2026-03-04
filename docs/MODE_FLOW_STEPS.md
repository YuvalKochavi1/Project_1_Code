# Mode-by-Mode Flow Steps

This is a practical flow map for each `mode` accepted by `analytic_wave_front_dispatch(...)`.

## Mode: `no_marshak`

Executed by `analytic_wave_front_no_marshak(...)`.

1. Normalize/sort times with `WavefrontHelpers.prepare_times`.
2. Compute constants (`eps`, `C0`, `pref`) via `WavefrontHelpers.compute_constants_for_wavefront`.
3. Build `Ts(t)` from `get_TD(...)`.
4. Build `H(t)=Ts^(4+alpha)`.
5. Integrate in time:
   - plain integral `∫H dt`, or
   - weighted integral when `lam_eff=True` because g changes in time in that case.
6. Compute front: `xF^2 = pref * C * H^(-eps) * integral`.
7. Restore original time order with `WavefrontHelpers.restore_original_order`.

Output:

- `xF`

---

## Mode: `marshak`

Executed by solver -> `_marshak_appendixA_march(..., wall_loss=False, ablation=False)`.

Per timestep:

1. Compute base Marshak flux.
2. Integrate energy-like term `E`.
3. Solve implicit `H_new` with Brent (`WavefrontHelpers.solve_for_H_new_brentq`).
4. Update `Ts`, `I`, and front `xF`.
5. Update heating-time map `t_heat`.

Output tuple:

- `xF_out, Ts_out, E_out, Ew_out, data_of_R, bessel_data`

---

## Mode: `marshak_wall_loss`

Executed by solver -> `_marshak_appendixA_march(..., wall_loss=True, ablation=False)`.

Adds to Marshak mode:

1. Compute wall loss each step via `WallLossModel.compute_wall_energy_loss`.
2. Subtract wall-loss term from flux.
3. Track cumulative wall energy (`E_wall_array_erg`).
4. Compute albedo diagnostics with `AlbedoModel.compute_albedo_step`.
5. Compute/store Bessel-related profiles in `bessel_data`.

Output tuple:

- `xF_out, Ts_out, E_out, Ew_out, data_of_R, bessel_data`

---

## Mode: `marshak_ablation`

Executed by solver -> `_marshak_appendixA_march(..., wall_loss=True, ablation=True)`.

Adds to wall-loss mode:

1. Get material-dependent `u_tilde` from CSV (`AblationModel.get_u_tilda_closest`).
2. Evolve radius profile `R_array` (`AblationModel.compute_R_t`).
3. Update cross-section area `A[i]` from `R_array[0]`.
4. Optionally update effective density (`AblationModel.compute_rho_effective`) when `vary_rho=True`.
5. Use updated geometry/density in `E`, `C`, and `xF` updates.

Output tuple:

- `xF_out, Ts_out, E_out, Ew_out, data_of_R, bessel_data`

---

## Notes on `_marshak_appendixA_march(...)` returned data

`bessel_data` is a dictionary keyed by time (ns). Each entry stores:

```
{
    'r_grid': r_grid.copy(),
    'z_grid': z.copy(),
    'J0_profiles': J0_profiles.copy(),
    'J0_profiles_approx': J0_profiles_approx.copy(),
    'kappa_0': kappa_0,
    'kappa_0_approx': kappa_0_approx,
    'epsilon': epsilon,
    'albedo': albedo,
    'lambda_ross': lambda_ross,
}
```

These entries are populated in the wall-loss branch (`wall_loss=True`, after the first few steps). In other modes, `bessel_data` is typically empty.

`data_of_R` is a dictionary mapping time in ns to the radius profile along `z` at that time:

```
{
    t_ns_0: R_profile_at_t0,
    t_ns_1: R_profile_at_t1,
    ...
}
```

Even without ablation, this structure is returned for consistency (initially holding the default radius profile).

`E_out` is the integrated Marshak energy term (total energy-like integral in the model output), and `Ew_out` is the cumulative wall-loss energy term. `Ew_out` is zero when wall loss is disabled.

## Notes on Units and Time

- Many routines internally convert between seconds and nanoseconds.
- `prepare_times` performs sorting and normalizes very large values.
- `data_of_R` keys are converted to ns at return time.

## Notes on Reusability

- Public API stays in `model_main.py`.
- Physics calculations are factored into class files:
  - `wavefront_helpers.py`
  - `wall_loss_model.py`
  - `ablation_model.py`
  - `albedo_model.py`
- Dispatch/facade is in `analytical_wavefront_solver.py`.
