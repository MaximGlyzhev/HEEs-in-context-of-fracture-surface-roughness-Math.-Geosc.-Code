import numpy as np

"""
This code implements the Higuchi Hurst exponent estimator on the fBm
model. 
"""


def higuchi_fd(profile, step_min=3, f=0.05, n=10):
    """
    Higuchi Fractal Dimension (HFD) with user-controlled fitting scales.

    Inputs
    -------
    profile : array-like
        1D time series / profile.
    step_min : int
        Minimal fitting scale (k_min). Always included.
    f : float
        Cut-off coefficient controlling maximal fitting scale:
        k_max is chosen strictly below L*f (and also <= L/2 for Higuchi).
    n : int
        Number of fitting points (number of k values).

    Outputs
    -------
    D : float
        Fractal dimension (Higuchi FD).
    H : float
        Hurst exponent, computed as 2 - D.
    """
    X = np.asarray(profile, dtype=float).ravel()
    if X.ndim != 1:
        raise ValueError("profile must be 1D.")
    L = X.size
    if L < 4:
        raise ValueError("profile is too short.")
        

    step_min = int(step_min)
    if step_min < 2:
        step_min = 2

    # k_max must be strictly below L*f, and Higuchi typically restricts k <= L/2
    k_max_by_f = int(np.floor(L * float(f))) - 1
    k_max = min(L // 2, k_max_by_f)

    if k_max < step_min:
        raise ValueError(
            f"Invalid scales: computed k_max={k_max} < step_min={step_min}. "
            f"Try increasing f or reducing step_min (profile length L={L})."
        )

    n = int(n)
    if n < 2:
        n = 2

    # --- Choose k values approximately equidistant on log scale, always include step_min ---
    k_candidates = np.arange(step_min, k_max + 1, dtype=int)
    log_min = np.log(k_candidates[0])
    log_max = np.log(k_candidates[-1])
    target_logs = np.linspace(log_min, log_max, n)

    # Always include step_min, then pick nearest distinct k for each target log
    chosen = [step_min]
    chosen_set = {step_min}
    for tl in target_logs:
        kt = int(np.round(np.exp(tl)))
        kt = max(step_min, min(k_max, kt))
        if kt not in chosen_set:
            chosen.append(kt)
            chosen_set.add(kt)

    # If rounding caused too few unique k's, fill by greedily adding k's that best match remaining targets
    if len(chosen) < n:
        remaining = [k for k in k_candidates if k not in chosen_set]
        remaining_logs = np.log(np.array(remaining, dtype=float))
        for tl in target_logs:
            if len(chosen) >= n or not remaining:
                break
            # nearest in log-space
            idx = int(np.argmin(np.abs(remaining_logs - tl)))
            k_add = remaining.pop(idx)
            remaining_logs = np.delete(remaining_logs, idx)
            chosen.append(k_add)
            chosen_set.add(k_add)

    k_arr = np.array(sorted(chosen))[:n]  # ensure sorted, cap at n

    # --- Higuchi curve length for each k (pure Python) ---
    N = L
    Lk = np.zeros_like(k_arr, dtype=float)

    for i, k in enumerate(k_arr):
        Lmk = 0.0
        for m in range(k):
            x_mk = X[m::k]
            if x_mk.size < 2:
                continue
            # number of intervals for this m,k
            num = (N - m - 1) // k
            if num <= 0:
                continue
            # Higuchi length for this m,k
            Lm = np.sum(np.abs(np.diff(x_mk))) * (N - 1) / (num * k)
            Lmk += Lm
        Lk[i] = (Lmk / k) / k  # average over m and normalize by k

    # Guard against any zeros (can happen for constant signals)
    if np.any(Lk <= 0):
        raise ValueError("Encountered non-positive curve lengths (signal may be constant).")

    # --- Fit slope on log-log scale: log(Lk) vs log(k); D = -slope ---
    lk = np.log2(k_arr.astype(float))
    lL = np.log2(Lk)
    slope = np.polyfit(lk, lL, 1)[0]
    D = -float(slope)
    H = 2.0 - D
    return D, H







