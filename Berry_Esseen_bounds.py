import numpy as np
import math
from typing import Iterable, Mapping, Tuple, Union, Optional

# Standard normal CDF using erf (no scipy needed)
def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Evaluates BEB for one estimator for a particular Hurst exponent H and profiel length L.
# A Hurst exponent are not an explicit input. 
# However, sigma and rho should be evaluated for a specific choice of H and L.

def berry_essen_bound(
    epsilon: float,
    p: float,
    sigma: float,
    rho: float,
    *,
    C: float = 0.4748,
    n_max: int = 10_000_000
) -> int:
    """
    Compute B(epsilon, p, sigma, rho) = smallest n in N such that
        1 - Phi(epsilon*sqrt(n)/sigma) + C*rho/(sqrt(n)*sigma^3) <= p/2

    Parameters
    ----------
    epsilon : float
        Error bound ε > 0.
    p : float
        Probability level p in (0,1).
    sigma : float
        Standard deviation σ > 0 of the estimator.
    rho : float
        Absolute third central moment ρ >= 0 of the estimator.
    C : float, default 0.4748
        Berry–Esseen constant used in your text.
    n_max : int
        Safety cap for search. If the bound is not met up to n_max, raises.

    Returns
    -------
    n : int
        Minimal n satisfying the inequality.
    """
    if not (epsilon > 0.0):
        raise ValueError("epsilon must be > 0.")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1).")
    if not (sigma > 0.0):
        raise ValueError("sigma must be > 0.")
    if not (rho >= 0.0):
        raise ValueError("rho must be >= 0.")
    if n_max < 1:
        raise ValueError("n_max must be >= 1.")

    target = p / 2.0

    def lhs(n: int) -> float:
        s = math.sqrt(n)
        z = (epsilon * s) / sigma
        term1 = 1.0 - _phi(z)
        term2 = (C * rho) / (s * (sigma ** 3))
        return term1 + term2

    # Quick check at n=1
    if lhs(1) <= target:
        return 1

    # Find an upper bracket by doubling n until condition is met
    lo, hi = 1, 2
    while hi <= n_max and lhs(hi) > target:
        lo, hi = hi, hi * 2

    if hi > n_max:
        # Try final check at n_max
        if lhs(n_max) > target:
            raise RuntimeError(
                f"Condition not met up to n_max={n_max}. "
                f"Try increasing n_max or check parameters."
            )
        hi = n_max

    # Binary search for minimal n in [lo+1, hi]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if lhs(mid) <= target:
            hi = mid
        else:
            lo = mid

    return hi

# Evluates maximum Berry-Essen bound for several Hurst exponents.
# Hurst exponents are not an explicit input. 
# However, sigma and rho should be evaluated for several Hurst exponents (and on specific profile length).
def universal_berry_essen_bound(
    epsilon: float,
    p: float,
    S: Iterable[float],
    sigma_by_H: Union[Mapping[float, float], np.ndarray],
    rho_by_H: Union[Mapping[float, float], np.ndarray],
    *,
    C: float = 0.4748,
    n_max: int = 10_000_000
) -> Tuple[int, float]:
    """
    Compute the S-universal Berry–Esseen bound:
        max_{H in S} B(epsilon, p, sigma(H), rho(H))

    Returns (B_universal, H_worst) where H_worst attains the max.

    sigma_by_H and rho_by_H can be:
    - dict-like mapping H -> value, OR
    - arrays aligned with S order.
    """
    S_list = list(S)
    if len(S_list) == 0:
        raise ValueError("S must be non-empty.")

    # Helper to fetch sigma/rho for each H
    def get_val(container, idx, H):
        if isinstance(container, Mapping):
            return float(container[H])
        else:
            return float(container[idx])

    worst_n = -1
    worst_H = None

    for i, H in enumerate(S_list):
        sigma = get_val(sigma_by_H, i, H)
        rho = get_val(rho_by_H, i, H)
        nH = berry_essen_bound(epsilon, p, sigma, rho, C=C, n_max=n_max)

        if nH > worst_n:
            worst_n = nH
            worst_H = H

    return worst_n, float(worst_H)



"""
This is a example input for  universal_berry_essen_bound

# parameters
epsilon = 0.02
p = 0.05

# Hurst set
S = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# example estimator statistics for ONE method and ONE length L
# (replace with your measured values)

sigma_by_H = {
    0.1: 0.085,
    0.2: 0.072,
    0.3: 0.061,
    0.4: 0.053,
    0.5: 0.048,
    0.6: 0.051,
    0.7: 0.058,
    0.8: 0.069,
    0.9: 0.083,
}

rho_by_H = {
    0.1: 0.0018,
    0.2: 0.0012,
    0.3: 0.0009,
    0.4: 0.0007,
    0.5: 0.0006,
    0.6: 0.0007,
    0.7: 0.0010,
    0.8: 0.0014,
    0.9: 0.0020,
}

B_universal, H_worst = universal_berry_essen_bound(
    epsilon=epsilon,
    p=p,
    S=S,
    sigma_by_H=sigma_by_H,
    rho_by_H=rho_by_H
)

print("Universal BEB:", B_universal)
print("Worst H:", H_worst)

"""






