# Code Flow Overview (`Project 1/model_main.py`)

This document explains how calls move through the analytical wavefront code after the class refactor.

## 1. Entry Points

Primary public APIs in `model_main.py`:

- `analytic_wave_front_no_marshak(...)`
- `analytic_wave_front_marshak(...)`
- `analytic_wave_front_marshak_gold_loss(...)`
- `analytic_wave_front_marshak_ablation(...)`
- `analytic_wave_front_dispatch(...)`

Most users call `analytic_wave_front_dispatch(...)` with a mode.

## 2. Default Solver Object (Lazy Singleton)

`model_main.py` defines:

- `_DEFAULT_ANALYTICAL_WAVEFRONT_SOLVER = None`
- `_get_default_solver()`

Behavior:

1. First call creates one `AnalyticalWavefrontSolver` instance.
2. It injects function pointers:
   - `no_marshak_fn=analytic_wave_front_no_marshak`
   - `march_fn=_marshak_appendixA_march`
3. Later calls reuse the same solver instance.

Why this exists:

- Single place to keep default wiring.
- Avoid re-instantiating solver repeatedly.
- Keeps API calls concise.

## 3. Dispatch Layer (`analytical_wavefront_solver.py`)

`AnalyticalWavefrontSolver` is a routing/facade class.

- `analytic_wave_front_dispatch(...)` selects mode.
- Mode methods call either:
  - `self.no_marshak_fn(...)` for no-Marshak, or
  - `self.march_fn(...)` for Marshak variants.

It does not implement the heavy numerics itself; it delegates to `model_main.py` core functions.

## 4. Core Physics Layers

### 4.1 Wavefront helper math

From `wavefront_helpers.py` (`WavefrontHelpers`):

- `prepare_times(...)`
- `compute_constants_for_wavefront(...)`
- `solve_for_H_new_brentq(...)`
- `restore_original_order(...)`

Used by both no-Marshak and Marshak paths.

### 4.2 Wall-loss model

From `wall_loss_model.py` (`WallLossModel`):

- `compute_wall_energy_loss(...)`
- `delta_e_vacuum_hJ_per_mm2(...)`
- `E_wall_gold(...)`, `E_wall_gold_dot(...)`, `E_wall_cupper(...)`, `E_wall_be(...)`

### 4.3 Ablation model

From `ablation_model.py` (`AblationModel`):

- `get_u_tilda_closest(...)`
- `ablation_velocity_gold(...)`, `ablation_velocity_cupper(...)`
- `compute_R_t(...)`
- `compute_rho_effective(...)`

### 4.4 Albedo model

From `albedo_model.py` (`AlbedoModel`):

- `compute_albedo_step(...)`
- `compute_albedo(...)`

## 5. Data Outputs by Path

### No Marshak path

`analytic_wave_front_no_marshak(...)` returns:

- `xF` (front position array)

### Marshak paths

`_marshak_appendixA_march(...)` returns:

- `xF_out`
- `Ts_out`
- `E_out`
- `Ew_out`
- `data_of_R`
- `bessel_data`

## 6. Important Internal Loop (Marshak)

Within `_marshak_appendixA_march(...)`, each timestep roughly does:

1. Compute incoming Marshak flux from `TD_now` and `Ts_prev`.
2. Optionally apply wall-loss correction.
3. Optionally apply ablation geometry and effective density updates.
4. Solve implicit equation for new `H` via Brent root solve.
5. Update `Ts`, integral `I`, and front `xF`.
6. Track heating times `t_heat` and optional Bessel/albedo diagnostics.

## 7. Quick Mental Model

Call chain for most runs:

`analytic_wave_front_dispatch`  -> `_get_default_solver` -> `AnalyticalWavefrontSolver.analytic_wave_front_dispatch` -> selected mode method -> `analytic_wave_front_no_marshak` or `_marshak_appendixA_march` -> helper/model classes (`WavefrontHelpers`, `WallLossModel`, `AblationModel`, `AlbedoModel`).
