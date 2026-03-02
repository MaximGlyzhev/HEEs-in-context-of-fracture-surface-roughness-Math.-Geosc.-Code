import numpy as np


def sva(profile, n_SVA=10, f_SVA=0.05, tau_min=3, d=1, detrend=False, nvals=None):
    """
    Semivariogram Analysis (SVA)
    to estimate the Hurst exponent from 1D profiles.

    Parameters
    ----------
    profile : array-like
        Samples y_1, ..., y_L of an equidistant profile.

        geometrically spaced (equidistant in log-scale), rounded down to ints.
    
    n_SVA : int, optional
        Number of scales / fitting points (default 10).
    f_SVA : float, optional
        Upper cut-off coefficient for maximum scale τ_n = f_SVA * L (default 0.05).
    tau_min : int, optional
        Minimal scale τ_1 (default 5; should be ≥ 2× resolution for real data).
    d : float, optional
        Spacing between consecutive profile points. Defaults to 1. Only affects
        the intercept (C_SVA), not the H estimate.
    detrend : bool, optional
        If True, linearly detrend the input profile before computing increments.
        (Recommended for natural fracture profiles.)
    nvals : array-like, optional
        Integer step sizes τ to use. If None, they are generated:
        τ_1 = tau_min (default 5), τ_n = floor(f_SVA * L),

    Returns
    -------
    H_SVA : float
        Estimated Hurst exponent.
    taus_used : ndarray of int
        The τ values actually used in the fit.
    I_tau : ndarray of float
        The second-moment increments I_τ for each τ in taus_used.
    C_SVA : float
        Estimated amplitude constant in I_τ ≈ C_SVA * (d τ)^{2 H_SVA}.
    """
    y = np.asarray(profile, dtype=float)
    L = y.size
    if L < 3:
        raise ValueError("Profile too short for SVA (need at least 3 samples).")

    # Optional linear detrend for natural profiles
    if detrend:
        x = np.arange(L, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
        y = y - trend

    # Default τ grid: geometric from tau_min to floor(f_SVA * L), log-equidistant
    if nvals is None:
        tau_max = max(int(np.floor(f_SVA * L)), tau_min)
        if n_SVA < 2 or tau_max <= tau_min:
            # Fallback to a small linear set if geometry is degenerate
            nvals = np.unique(np.clip(np.arange(tau_min, tau_min + n_SVA), 1, L - 1))
        else:
            a = (tau_max / float(tau_min)) ** (1.0 / (n_SVA - 1))
            raw = tau_min * (a ** np.arange(n_SVA))
            nvals = np.unique(np.floor(raw).astype(int))

    # Keep only valid τ with at least one increment
    nvals = np.array([t for t in nvals if 1 <= t < L], dtype=int)
    if nvals.size < 2:
        raise ValueError("Not enough valid τ values for regression. Adjust tau_min/f_SVA/n_SVA.")

    # Compute I_τ = (1/(L-τ)) * sum_{k=1}^{L-τ} (y_{k+τ} - y_k)^2
    I_tau = np.empty(nvals.size, dtype=float)
    for i, tau in enumerate(nvals):
        diffs = y[tau:] - y[:-tau]
        I_tau[i] = np.mean(diffs * diffs)

    # Log-log linear regression: log I_τ = log C_SVA + 2 H_SVA * log(d τ)
    x = np.log(d * nvals.astype(float))
    ylog = np.log(I_tau)
    slope, intercept = np.polyfit(x, ylog, 1)

    H_SVA = 0.5 * slope
    C_SVA = float(np.exp(intercept))

    return H_SVA, C_SVA, I_tau


