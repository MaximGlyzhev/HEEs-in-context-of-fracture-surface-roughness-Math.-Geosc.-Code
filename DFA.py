import numpy as np

"""
This code implements the detrended fluctuation analysis Hurst exponent
estimator adjusted the fBm model. The inputs are assumed to be fBm profiles. 
A summation/integration step is ommited since fBm profiles are already assumed 
to be integrated fGn profiles.
This code can also be applied on fracture surface profiles.
"""


def dfa(profile, order=1, l_DFA=10, f_DFA=0.1, n_DFA=20, nvals=None):
    """
    Perform Detrended Fluctuation Analysis (DFA) to estimate the Hurst exponent.

    Parameters:
        profile (array-like): The input signal or profile (length L)
        order (int): Polynomial order for detrending (default 1)
        l_DFA (int): Minimal scale (default 20)
        f_DFA (float): Max scale fraction of profile length (default 0.1)
        n_DFA (int): Number of scales/fitting points (default 20)
        nvals (array-like, optional): Scales to use. If None, computed as per DFA description.

    Returns:
        H (float): Estimated Hurst exponent
        nvals (array): The scales used
        F_n (array): The fluctuation function values
    """
    signal = np.array(profile)
    N = len(signal)

    # Compute default nvals according to the theoretical description
    if nvals is None:
        a = (N * f_DFA / l_DFA) ** (1 / (n_DFA - 1))
        nvals = l_DFA * a ** np.arange(n_DFA)
        nvals = np.floor(nvals).astype(int)

    F_n = []
    for n in nvals:
        segments = int(np.floor(N / n))
        F_n_segment = []
        for i in range(segments):
            indices = np.arange(i * n, (i + 1) * n)
            coeffs = np.polyfit(indices, signal[indices], order)
            trend = np.polyval(coeffs, indices)
            F_n_segment.append(np.sqrt(np.mean((signal[indices] - trend) ** 2)))
        F_n.append(np.mean(F_n_segment))

    F_n = np.array(F_n)
    coeffs = np.polyfit(np.log(nvals), np.log(F_n), 1)

    H = coeffs[0]
    return H, nvals, F_n




