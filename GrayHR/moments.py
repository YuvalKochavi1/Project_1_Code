"""
Moments calculation module for the Unified Gray HR Model.

Implements:
1. Gray storage factor G(R) via exact Gauss hypergeometric function 2F1.
2. Profile moments M_p(t, z) and N_p(t, z) via high-order Gauss-Legendre quadrature
   on [0, 1] for ultra-fast, machine-precision vector evaluation.
"""

import numpy as np
from scipy.special import hyp2f1
from scipy.integrate import quad

# Precompute 40-point Gauss-Legendre nodes and weights on [0, 1]
_nodes_raw, _weights_raw = np.polynomial.legendre.leggauss(40)
_GL_NODES = 0.5 * (_nodes_raw + 1.0)
_GL_WEIGHTS = 0.5 * _weights_raw


def compute_G(R: float, alpha: float, beta: float) -> float:
    """
    Computes the gray storage factor G(R) using the exact hypergeometric form:
    
        G(R) = 2F1(1, (4+alpha-beta)/(4-beta); 1 + (4+alpha-beta)/(4-beta); -R)
        
    where R = (a_rad * T_s^4) / (f * rho^(1-mu) * T_m^beta).
    """
    if R <= 0.0:
        return 1.0
    
    b = (4.0 + alpha - beta) / (4.0 - beta)
    c = 1.0 + b
    
    val = float(hyp2f1(1.0, b, c, -R))
    return max(0.0, val)


def compute_G_quad(R: float, alpha: float, beta: float) -> float:
    """
    Evaluates G(R) via direct quadrature of its defining integral:
    
        G(R) = (4+alpha-beta) * int_0^1 (xi^(3+alpha-beta) / (1 + R * xi^(4-beta))) dxi
    """
    if R <= 0.0:
        return 1.0
    
    prefactor = 4.0 + alpha - beta
    exp_num = 3.0 + alpha - beta
    exp_den = 4.0 - beta
    
    def integrand(xi):
        return (xi ** exp_num) / (1.0 + R * (xi ** exp_den))
        
    val, _ = quad(integrand, 0.0, 1.0, epsrel=1e-10, epsabs=1e-10)
    return max(0.0, prefactor * val)


def compute_M_p(B: float, z: float, p: float, front_exponent: float) -> float:
    """
    Computes profile moment M_p(B, z):
    
        M_p(B, z) = int_0^1 [(1-y)(1 + B*y)]^(p * front_exponent) * exp(-z*y) dy
        
    Evaluated using 40-point Gauss-Legendre quadrature (`_GL_NODES`) for maximum speed.
    """
    B_clamped = max(-0.99999, float(B))
    exp_factor = p * front_exponent
    
    base = (1.0 - _GL_NODES) * (1.0 + B_clamped * _GL_NODES)
    base = np.maximum(0.0, base)
    exp_arg = np.clip(-z * _GL_NODES, -100.0, 100.0)
    
    integrand = (base ** exp_factor) * np.exp(exp_arg)
    return float(np.dot(_GL_WEIGHTS, integrand))


def compute_N_p(B: float, z: float, p: float, front_exponent: float) -> float:
    """
    Computes profile moment N_p(B, z):
    
        N_p(B, z) = int_0^1 y * [(1-y)(1 + B*y)]^(p * front_exponent) * exp(-z*y) dy
        
    Evaluated using 40-point Gauss-Legendre quadrature (`_GL_NODES`) for maximum speed.
    """
    B_clamped = max(-0.99999, float(B))
    exp_factor = p * front_exponent
    
    base = (1.0 - _GL_NODES) * (1.0 + B_clamped * _GL_NODES)
    base = np.maximum(0.0, base)
    exp_arg = np.clip(-z * _GL_NODES, -100.0, 100.0)
    
    integrand = _GL_NODES * (base ** exp_factor) * np.exp(exp_arg)
    return float(np.dot(_GL_WEIGHTS, integrand))
