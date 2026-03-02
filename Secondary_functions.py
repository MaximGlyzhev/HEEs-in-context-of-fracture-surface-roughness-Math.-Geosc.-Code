import numpy as np
import math
from typing import Tuple, Optional


"""
This code contains supplementary function to generated fBms with displacements.
This code is not interesting for practial applications

"""

def _normal_cdf(x: float, sigma: float) -> float:
    """CDF of N(0, sigma^2) using erf (no scipy)."""
    if sigma <= 0:
        return 0.0 if x < 0 else 1.0
    return 0.5 * (1.0 + math.erf(x / (sigma * math.sqrt(2.0))))


def _truncated_bin_prob(a: float, b: float, M: int, sigma: float) -> float:
    """
    Probability P(Z in [a,b] | Z in [-(M+0.5), M+0.5]) for Z ~ N(0,sigma^2).
    """
    lo = -(M + 0.5)
    hi = +(M + 0.5)

    num = _normal_cdf(b, sigma) - _normal_cdf(a, sigma)
    den = _normal_cdf(hi, sigma) - _normal_cdf(lo, sigma)
    # den > 0 for sigma>0; for completeness:
    return num / den if den > 0 else 0.0


def _sigma_from_pnd(p_nd: float, M: int, tol: float = 1e-12) -> float:
    """
    Find sigma so that truncated-bin probability for k=0 bin [-0.5,0.5]
    equals p_nd, i.e. P(|Z|<=0.5 | Z in [-(M+0.5), M+0.5]) = p_nd.

    Uses monotone binary search: as sigma increases, mass spreads -> p_nd decreases.
    """
    if not (0.0 < p_nd < 1.0):
        raise ValueError("p_nd must be in (0,1).")

    # For sigma -> 0, p0 -> 1
    # For sigma -> inf, distribution ~ uniform-ish over truncation, so p0 -> 1/(2M+1)
    pmin = 1.0 / (2 * M + 1)
    if p_nd < pmin - 1e-15:
        raise ValueError(
            f"p_nd is too small for M={M}. Minimum achievable is about 1/(2M+1)={pmin:.6g}."
        )

    if abs(p_nd - 1.0) < 1e-15:
        return 0.0  # essentially no displacement

    # Bracket sigma
    lo = 1e-12
    hi = 1.0
    def p0(sig: float) -> float:
        return _truncated_bin_prob(-0.5, 0.5, M=M, sigma=sig)

    # Increase hi until p0(hi) <= p_nd (or until hi huge)
    while p0(hi) > p_nd and hi < 1e6:
        hi *= 2.0

    # Binary search
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = p0(mid)
        if abs(val - p_nd) < tol:
            return mid
        # val > p_nd means sigma too small (too concentrated), so increase sigma
        if val > p_nd:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def displacement_probabilities_from_pnd(p_nd: float, M: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build discrete displacement distribution over k = [-M,...,M]
    from a truncated normal, with p(k=0) = p_nd.

    Returns
    -------
    ks : np.ndarray shape (2M+1,)
        Integer displacements.
    pk : np.ndarray shape (2M+1,)
        Probabilities summing to 1.
    sigma : float
        The underlying normal stddev that matches p_nd after truncation+binning.
    """
    if not isinstance(M, int) or M < 1:
        raise ValueError("M must be an integer >= 1.")

    sigma = _sigma_from_pnd(p_nd=p_nd, M=M)

    ks = np.arange(-M, M + 1, dtype=int)
    pk = np.empty(ks.size, dtype=float)

    # Compute binned probs within truncation and renormalize (should already sum to 1)
    for i, k in enumerate(ks):
        a = k - 0.5
        b = k + 0.5
        pk[i] = _truncated_bin_prob(a, b, M=M, sigma=sigma)

    pk = pk / pk.sum()
    return ks, pk, sigma

