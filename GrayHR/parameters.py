"""
Parameters module for the Unified Gray Hammer-Rosen (HR) Model.

This module provides data classes and material presets for the 
two-temperature gray Marshak wave model described in:
'A Unified Gray Hammer-Rosen Model for Surface- and Bath-Temperature Drives'.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GrayHRParameters:
    """
    Physical and material parameters for the Unified Gray HR Model.
    All parameters are assumed to be in consistent CGS/eV units unless noted.
    """
    # Fundamental constants
    c: float = 2.99792458e10         # Speed of light [cm/s]
    sigma_sb: float = 5.670374419e-5 # Stefan-Boltzmann constant [erg/cm^2/s/K^4]
    
    # Radiation constant a_rad = 4 * sigma_sb / c
    # Note: If temperatures are in eV, a_rad should be in erg/cm^3/eV^4.
    # 1 eV = 11604.51812 K
    a_rad: float = 1.372e2           # Radiation constant in [erg/cm^3/eV^4]
    
    # Material energy equation: U = f * rho^(1-mu) * T^beta [erg/cm^3]
    f: float = 8.77e13               # Material specific heat scale [erg/g/eV^beta]
    beta: float = 1.1                # Material temperature exponent
    mu: float = 0.09                 # Density exponent for material energy
    rho: float = 0.05                # Material density [g/cm^3]
    
    # Rosseland opacity power law: 1 / kappa_R = g * T^alpha * rho^(-lambda)
    g: float = 1.0 / 9175.0          # Rosseland opacity scale
    alpha: float = 3.53              # Rosseland temperature exponent
    lambda_param: float = 0.75       # Rosseland density exponent
    
    # Absorption opacity power law: 1 / kappa_a = g' * T^(alpha') * rho^(-lambda')
    # If None, defaults to Rosseland parameters (kappa_a = kappa_R)
    g_prime: Optional[float] = None
    alpha_prime: Optional[float] = None
    lambda_prime: Optional[float] = None

    def __post_init__(self):
        # Default absorption parameters to Rosseland parameters if not provided
        if self.g_prime is None:
            self.g_prime = self.g
        if self.alpha_prime is None:
            self.alpha_prime = self.alpha
        if self.lambda_prime is None:
            self.lambda_prime = self.lambda_param

    @property
    def eps(self) -> float:
        """Standard Hammer-Rosen epsilon parameter: eps = beta / (4 + alpha)."""
        return self.beta / (4.0 + self.alpha)

    @property
    def C(self) -> float:
        """Standard Hammer-Rosen C parameter."""
        # C = (4 * a_rad * c * g) / (3 * (4 + alpha) * f) * rho^(-2 + mu - lambda)
        numerator = 4.0 * self.a_rad * self.c * self.g
        denominator = 3.0 * (4.0 + self.alpha) * self.f
        rho_factor = self.rho ** (-2.0 + self.mu - self.lambda_param)
        return (numerator / denominator) * rho_factor

    @property
    def front_exponent(self) -> float:
        """Leading front profile exponent: 1 / (4 + alpha - beta)."""
        return 1.0 / (4.0 + self.alpha - self.beta)

    @classmethod
    def from_preset(cls, material_name: str, **overrides) -> "GrayHRParameters":
        """
        Factory method to load parameters from known experimental presets.
        
        Supported presets: 'SiO2', 'Gold', 'C11H16Pb0.3852', 'C6H12'
        """
        presets = {
            "SiO2": dict(
                f=8.77e13,
                g=1.0 / 9175.0,
                alpha=3.53,
                beta=1.1,
                lambda_param=0.75,
                mu=0.09,
                rho=0.05,
            ),
            "Gold": dict(
                f=1.5e13,
                g=1.0 / 5000.0,
                alpha=1.5,
                beta=1.2,
                lambda_param=0.2,
                mu=0.0,
                rho=0.1,
            ),
            "C11H16Pb0.3852": dict(
                f=10.17e13,
                g=1.0 / 3200.0,
                alpha=1.57,
                beta=1.2,
                lambda_param=0.1,
                mu=0.0,
                rho=0.08,
            ),
            "C6H12": dict(
                f=12.27e13,
                g=1.0 / 3926.6,
                alpha=2.98,
                beta=1.0,
                lambda_param=0.95,
                mu=0.04,
                rho=0.05,
            ),
        }
        
        if material_name not in presets:
            raise ValueError(f"Unknown preset '{material_name}'. Available: {list(presets.keys())}")
        
        kwargs = presets[material_name].copy()
        kwargs.update(overrides)
        return cls(**kwargs)
