# 2D Simulation Flow (Full Pipeline)

This document explains the end-to-end execution flow of the split 2D simulation code under `2D_simulation/`.

## 1. Files and Responsibilities

- `run_2D_simulation.py`
  - Thin runner script.
  - Calls the default pipeline.

- `simulation_2d_pipeline.py`
  - High-level orchestration.
  - Creates configured simulation, runs time loop, saves run data, calls plotting/export.

- `simulation_2d_core.py`
  - Physics and numerics core.
  - Grid generation, transport/coupling coefficients, implicit solve, time stepping.

- `simulation_2d_plots.py`
  - Figure generation and figure-data exports.
  - Saves PNG figures and CSV data products.

- `csv_helpers.py`
  - Shared filesystem/data export helpers (`ensure_dir`, `save_series_csv`).

## 2. Execution Entry

Typical run path:

1. `python 2D_simulation/run_2D_simulation.py`
2. Runner imports `run_default_pipeline(...)` from `simulation_2d_pipeline.py`.
3. `run_default_pipeline(...)` executes the full pipeline.

## 3. Pipeline Flow (`simulation_2d_pipeline.py`)

Inside `run_default_pipeline(...)`:

1. `create_simulation(...)`
  - Builds material parameters and geometry.
  - Loads drive temperature curve (`T_drive.csv`).
  - Returns `SelfSimilarDiffusion2D` instance.

2. `run_simulation(...)`
  - Builds `times_to_store`.
  - Calls `sim.run(...)` from core.
  - Returns `stored_t`, `stored_Um`, `stored_Tm`, `stored_TR`.

3. Save full run arrays
  - `save_run_data(...)` writes `Data_new/<Experiment>/<Material>/2D/run_outputs.npz`.

4. Postprocessing signal
  - Computes centerline front with `sim.compute_front_at_r(...)`.

5. Plot/export calls
  - `plot_temperature_maps_gouraud(...)`
  - `plot_temperature_maps_simple(...)`
  - `plot_front_vs_time(...)`
  - `plot_front_surface(...)`
  - `plot_energy_comparison(...)`

## 4. Core Solver Flow (`simulation_2d_core.py`)

Main numerical sequence:

1. Initialize state arrays
  - Radiation energy `E` and material energy proxy `UR`.

2. Time loop (`run(...)`)
  - Adjust dt to hit store times.
  - Call `implicit_step(...)` for each step.
  - Compute/store `Um`, `Tm`, `TR` when hitting target times.
  - Adapt dt by relative-change controller.

3. Implicit solve (`implicit_step(...)`)
  - Build lagged coefficients from current state.
  - Build sparse linear system.
  - Solve with direct or iterative linear solver.
  - Reconstruct full `E` and update `UR` implicitly.

4. Derived diagnostics methods
  - `compute_front_at_r(...)`
  - `compute_front_surface(...)`
  - `compute_energy_foam(...)`
  - `compute_energy_gold(...)`

## 5. Plot and Export Flow (`simulation_2d_plots.py`)

### A. Temperature maps (gouraud)

- Figure: `Figures_new/<Experiment>/<Material>/2D_simulation/Tmap_<t>ns.png`
- Data CSV:
  - Cell data: `Data_new/<Experiment>/<Material>/2D_simulation/temperature_maps_gouraud/Tmap_<t>ns.csv`
  - Vertex data: `Data_new/<Experiment>/<Material>/2D_simulation/temperature_maps_gouraud/Tmap_<t>ns_vertex.csv`

### B. Temperature maps (simple)

- Figure: `Figures_new/<Experiment>/<Material>/2D_simulation/heatmap_<t>ns.png`
- Data CSV:
  - `Data_new/<Experiment>/<Material>/2D_simulation/temperature_maps_simple/heatmap_<t>ns.csv`

### C. Front vs time

- Figure: `Figures_new/<Experiment>/<Material>/2D_simulation/front_position - Front Position vs Time at r=0.png`
- Data CSV:
  - `Data_new/<Experiment>/<Material>/2D_simulation/front_vs_time/front_position_vs_time_r0.csv`

### D. Front surface

- Figure: `Figures_new/<Experiment>/<Material>/2D_simulation/front_surface - Front Surface zF vs r.png`
- Data CSV:
  - `Data_new/<Experiment>/<Material>/2D_simulation/front_surface/front_surface_profiles.csv`

### E. Energy comparison

- Figure: `Figures_new/<Experiment>/<Material>/2D_simulation/energy_comparison - Foam Energy vs Time.png`
- Data CSV:
  - `Data_new/<Experiment>/<Material>/2D_simulation/energy_comparison/simulated_energy_vs_time.csv`

## 6. Import Path Notes

The split modules add project root to `sys.path` at runtime when needed. This allows:

- Running from project root.
- Running scripts located inside `2D_simulation/` while still importing top-level helpers like `csv_helpers.py`.

## 7. Quick Call Graph

`run_2D_simulation.py`
-> `simulation_2d_pipeline.run_default_pipeline(...)`
-> `create_simulation(...)`
-> `simulation_2d_core.SelfSimilarDiffusion2D(...)`
-> `run_simulation(...)`
-> `SelfSimilarDiffusion2D.run(...)`
-> `SelfSimilarDiffusion2D.implicit_step(...)`
-> save run arrays (`run_outputs.npz`)
-> plotting/export functions in `simulation_2d_plots.py`
-> PNG + CSV outputs in `Figures_new/.../2D_simulation` and `Data_new/.../2D_simulation`
