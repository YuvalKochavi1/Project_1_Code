"""
GrayHR: Unified Gray Hammer-Rosen (HR) Model Package.

Implements the self-contained 7-step recipe from Section 10 of:
'A Unified Gray Hammer-Rosen Model for Surface- and Bath-Temperature Drives'.

Exports:
    - GrayHRParameters: Physical and material property parameters container.
    - SurfaceDriveSolver: Solver for prescribed surface temperature T_s(t).
    - BathDriveSolver: Solver for prescribed bath temperature T_bath(t).
    - GrayHRSolution: Solution container with profile evaluation capabilities.
"""

from parameters import GrayHRParameters
from solvers import SurfaceDriveSolver, BathDriveSolver, GrayHRSolution
from moments import compute_G, compute_M_p, compute_N_p

__all__ = [
    "GrayHRParameters",
    "SurfaceDriveSolver",
    "BathDriveSolver",
    "GrayHRSolution",
    "compute_G",
    "compute_M_p",
    "compute_N_p",
]
