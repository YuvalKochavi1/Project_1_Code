"""
Compatibility entrypoint for the split 1D simulation modules.

This file keeps the old run command working while the code now lives in:
- simulation.py
- simulation_1d_pipeline.py
"""

from simulation import *
from simulation_1d_pipeline import *


if __name__ == "__main__":
    results = run_default_pipeline(material=Material)
    print(f"Completed {len(results['stored_t'])} stored snapshots.")
    print(f"Stored outputs in: {results['data_dir']}")
    print(f"Stored figures in: {results['figures_dir']}")
    print(
        f"Front position range: {results['front_positions'].min():.6g} "
        f"to {results['front_positions'].max():.6g} cm"
    )
    print(
        f"Total energy range: {results['total_energies'].min():.6g} "
        f"to {results['total_energies'].max():.6g} hJ/mm^2"
    )
