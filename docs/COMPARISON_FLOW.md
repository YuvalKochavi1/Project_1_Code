# Comparison Module Flow (`Project 1/comparison.py`)

This file maps the execution flow of the plotting/comparison driver.

## 1. Main Entry

When run directly:

1. Build `times_to_store` in `__main__`.
2. Call selected comparison/diagnostic functions (currently `plot_both_marshak_and_nonmarshak_heat_fronts(...)` and `plot_albedo_z0_vs_time(...)` are active).
3. Each function computes analytics and/or loads stored CSV data.
4. Figures are rendered and saved through `save_figure(...)`.

## 2. Data Sources Used by Comparison Routines

Comparison functions combine:

- Simulation outputs from `Project 1/data/*.csv`.
- Article/reference CSVs via helpers from `csv_helpers.py` (`article_front_path`, `article_temperature_path`, `article_energy_path`).
- Analytical curves from `analytic_wave_front_dispatch(...)` in `model_main.py`.

## 3. Common Function Pattern (Most `compare_with_*` Functions)

Typical flow inside one comparison function:

1. Request one or more analytic series with `analytic_wave_front_dispatch(...)` using different `mode` values.
2. Create figure (`plt.figure(...)`).
3. Plot analytic series (direct `plt.plot(...)` and/or helper wrappers).
4. Read article/simulation CSVs and overlay curves/points.
5. Set labels, limits, title, legend, and layout.
6. Persist output via `save_figure("...", model1_5=True|model2_D=True)`.

## 4. Simulation Pipeline Flow

For full run-and-plot pipeline, `simulate()` calls:

1. `run_case(...)`
   - `init_state()`
   - `run_time_loop(...)`
   - write stored arrays to CSV in `Project 1/data`.
2. `plot_front_positions_and_energies(...)`
   - read back CSV snapshots
   - compute front/energy
   - call:
     - `plot_temperature_profiles(...)`
     - `plot_front_positions(...)`
     - `plot_energies(...)`

## 5. 2D Diagnostics Flow

### `plot_both_marshak_and_nonmarshak_heat_fronts(...)`

1. Load stored Marshak and non-Marshak simulation snapshots.
2. Build multiple analytic fronts (Marshak, no-Marshak, wall-loss, ablation variants).
3. Plot front-position comparison against article data.
4. Save front plot.
5. Plot and save temperature comparison (`T_s` + drive temperature).
6. If Bessel data exists, call:
   - `plot_2D_front_spatial(...)`
   - `plot_temperature_heatmap_2D(...)`

### `plot_albedo_z0_vs_time(...)`

1. Call `analytic_wave_front_dispatch(...)` for a wall-loss/ablation mode.
2. Extract `bessel_data`.
3. Build `albedo(t)` from dictionary entries.
4. Plot and save albedo-vs-time figure.

## 6. Key Output Artifacts

- Front position comparison images.
- Temperature comparison images.
- Total energy images.
- Optional 2D spatial front and temperature heatmaps.

All outputs are routed through `save_figure(...)` to the current figures directory configuration.
