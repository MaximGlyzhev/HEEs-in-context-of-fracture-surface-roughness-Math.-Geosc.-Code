import numpy as np
from scipy.signal import welch


def hurst_psd_welch_text(
    profile: np.ndarray,
    l_Welch: int | None = None,
    O_Welch: int | None = None,
    F_Welch: int | None = None,
    fs: float = 1.0,
    window: str = "hann",
    detrend: str = "constant",
    average: str = "median",
    scaling: str = "density",
    fmin: float | None = None,
    fmax: float = 0.05,
) -> float:
    """
    PSD-based Hurst exponent estimator using SciPy's Welch method,
    with parameters corresponding to the text:

        l_Welch = floor(L/2)   (segment length)
        O_Welch = floor(L/4)   (overlap)
        F_Welch = L            (FFT length)

    Parameters
    ----------
    profile : np.ndarray
        1D profile (e.g., fractional Brownian motion).

    l_Welch : int, optional
        Segment length used by Welch (maps to scipy.signal.welch `nperseg`).
        Default: floor(L/2).

    O_Welch : int, optional
        Overlap between segments (maps to welch `noverlap`).
        Default: floor(L/4).

    F_Welch : int, optional
        FFT length (maps to welch `nfft`).
        Default: L.

    fs : float, optional
        Sampling frequency of the profile. Only affects frequency axis scaling.
        Default: 1.0.

    window : str or tuple or array_like, optional
        Window applied to each segment before FFT.

        Common string options:
            "hann" (default) — good general-purpose spectral estimation
            "hamming"
            "blackman"
            "boxcar"      (no window / rectangular)
            "bartlett"
            "flattop"
            "parzen"
            "bohman"
            "nuttall"
            "cosine"

        You may also pass:
            - A tuple (window_name, param) for parameterized windows
              e.g. ("kaiser", beta)
            - A custom array of length l_Welch

        See: scipy.signal.get_window

    detrend : str or function or False, optional
        Specifies how each segment is detrended before PSD computation.

        Options:
            "constant" (default)
                Removes mean from each segment.
                Recommended for most PSD/Hurst estimation.

            "linear"
                Removes linear trend from each segment.

            False
                No detrending applied.

            callable
                Custom detrend function applied to each segment.

    average : {"mean", "median"}, optional
        Method used to average periodograms across segments.

        "mean"   — standard Welch averaging (default in SciPy)
        "median" — more robust to outliers/spikes in PSD (recommended for fBm)

        Default here: "median"

    scaling : {"density", "spectrum"}, optional
        Defines scaling of PSD output.

        "density" (default)
            Power spectral density (PSD) in units power/Hz.
            Recommended for slope fitting and Hurst estimation.

        "spectrum"
            Power spectrum (no division by bandwidth).

        Hurst estimation typically uses "density".

    fmin : float, optional
        Minimum frequency used for log-log slope fitting.
        If None → default = 3/L (low-frequency cutoff).

    fmax : float, optional
        Maximum frequency used for log-log slope fitting.
        Should remain in low-frequency scaling region.
        Default: 0.05.

    Returns
    -------
    H : float
        Estimated Hurst exponent using PSD slope:

            S(f) ~ f^(-2H-1)
            => H = -(slope + 1)/2
    """
    L = len(profile)

    # --- defaults exactly as stated in the text
    if l_Welch is None:
        l_Welch = L // 2
    if O_Welch is None:
        O_Welch = L // 4
    if F_Welch is None:
        F_Welch = L

    l_Welch = int(l_Welch)
    O_Welch = int(O_Welch)
    F_Welch = int(F_Welch)

    # --- sanity checks
    if not (2 <= l_Welch <= L):
        raise ValueError(f"l_Welch must be in [2, L]; got {l_Welch} with L={L}.")
    if not (0 <= O_Welch < l_Welch):
        raise ValueError(f"O_Welch must satisfy 0 <= O_Welch < l_Welch; got {O_Welch}, {l_Welch}.")
    if F_Welch < l_Welch:
        raise ValueError(f"F_Welch must be >= l_Welch; got {F_Welch} < {l_Welch}.")

    freqs, psd = welch(
        profile,
        fs=fs,
        window=window,
        nperseg=l_Welch,
        noverlap=O_Welch,
        nfft=F_Welch,
        detrend=detrend,
        average=average,
        scaling=scaling,
    )

    if fmin is None:
        fmin = 4 / L

    mask = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[mask]
    p = psd[mask]

    good = (f > 0) & (p > 0)
    f = f[good]
    p = p[good]

    if len(f) < 2:
        raise RuntimeError("Not enough frequency points in the fit range. Adjust fmin/fmax or profile length.")

    slope, _ = np.polyfit(np.log(f), np.log(p), 1)

    # PSD slope relation for fBm
    H = -(slope + 1) / 2
    return float(H)
