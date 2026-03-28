"""
Compatibility entrypoint for the split 2D simulation modules.

This file keeps the old run command working while the code now lives in:
- simulation_2d_core.py
- simulation_2d_plots.py
- simulation_2d_pipeline.py
"""

from simulation_2d_core import *
from simulation_2d_plots import *
from simulation_2d_pipeline import *


if __name__ == "__main__":
    run_default_pipeline(material=Material)
