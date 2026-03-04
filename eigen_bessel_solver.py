import numpy as np
from scipy import special
from scipy.optimize import brentq

def f_x(x: float, eps: float) -> float:
    """f(x) = x J1(x) - eps J0(x). Roots x_n give kappa_n = x_n / R."""
    return x * special.j1(x) - eps * special.j0(x)

def find_roots_x(eps: float,
                 n_roots: int,
                 x_max: float = None,
                 dx: float = 1e-2) -> np.ndarray:
    """
    Find the first n_roots positive roots x_n of x J1(x) = eps J0(x).
    Bracket by scanning with step dx, then use brentq in each bracket.
    """
    if n_roots <= 0:
        return np.array([])

    # Heuristic x_max if not provided: roots are ~ (n + 1/4)pi
    if x_max is None:
        x_max = (n_roots + 2.0) * np.pi

    # Scan for sign changes
    xs = np.arange(dx, x_max + dx, dx)  # start at dx to skip x=0
    fs = f_x(xs, eps)

    roots = []
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        fa, fb = fs[i], fs[i + 1]

        # Skip NaNs/infs
        if not np.isfinite(fa) or not np.isfinite(fb):
            continue

        # Exact hit (rare)
        if fa == 0.0:
            roots.append(a)
        # Sign change -> bracketed root
        elif fa * fb < 0.0:
            root = brentq(f_x, a, b, args=(eps,), maxiter=200)
            roots.append(root)

        if len(roots) >= n_roots:
            break

    if len(roots) < n_roots:
        raise RuntimeError(
            f"Found only {len(roots)} roots up to x_max={x_max}. "
            f"Increase x_max or decrease dx."
        )

    return np.array(roots[:n_roots])

def kappa_roots(eps: float, R: float, n_roots: int,
                x_max: float = None, dx: float = 1e-2) -> np.ndarray:
    """Return kappa_n = x_n / R for the first n_roots."""
    x_roots = find_roots_x(eps, n_roots, x_max=x_max, dx=dx)
    return x_roots / R

if __name__ == "__main__":
    eps = 0.05   # <-- set your epsilon
    R   = 0.08  # <-- set your R (same units you want for kappa^-1)
    N   = 3    # number of roots

    k_n = kappa_roots(eps, R, N, dx=1e-3)
    print(k_n)
    
    print("n    x_n (= kappa_n * R)          kappa_n      J_0(x_n) ")
    for i, kv in enumerate(k_n, start=1):
        print(f"{i:2d}   {kv * R: .12f}   {kv: .12f}   {special.j0(kv * R): .12f}")
