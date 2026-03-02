import Secondary_functions
import numpy as np
import math
from fBm_circulant_embedding import fBm_on_unit_interval
from typing import Tuple, Optional

def generate_fbm_with_x_displacements(
    H: float,
    L: int,
    topothesy: float,
    M: int,
    p_nd: float,
    rng: Optional[np.random.Generator] = None,
    return_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate an fBm profile on [0,1] of length L, with x-axis displacements modeled
    as discrete shifts within +/- M fine-grid neighbors.

    Procedure:
    - Generate fine fBm on [0,1] with Nfine = (2M+1)*L points.
    - Non-displaced "true" indices: M + n*(2M+1), n=0..L-1
    - Displace each by integer k in [-M,M] with truncated-normal binned probs
      determined by p_nd = P(k=0).

    Returns
    -------
    x_disp : np.ndarray shape (L,)
        Displaced x-coordinates in [0,1].
    h_disp : np.ndarray shape (L,)
        Heights at displaced positions.

    If return_debug=True, also returns a dict with internal details.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(L, int) or L < 2:
        raise ValueError("L must be an integer >= 2.")
    if not isinstance(M, int) or M < 1:
        raise ValueError("M must be an integer >= 1.")
    if not (0.0 < p_nd < 1.0):
        raise ValueError("p_nd must be in (0,1).")

    step = 2 * M + 1
    Nfine = step * L

    # fine-grid fBm on unit interval (your provided function)
    fine_h = fBm_on_unit_interval(H=H, N=Nfine, topothesy=topothesy, rng=rng)
    fine_x = np.linspace(0.0, 1.0, Nfine)

    base_idx = M + np.arange(L) * step  # indices of the non-displaced profile
    ks, pk, sigma = Secondary_functions.displacement_probabilities_from_pnd(p_nd=p_nd, M=M)


    # sample displacements for each point
    disp = rng.choice(ks, size=L, p=pk)
    disp_idx = base_idx + disp  # always in-bounds by construction

    x_disp = fine_x[disp_idx]
    h_disp = ((L-1)**H)*fine_h[disp_idx] # Rescalinng to true profile length

    if not return_debug:
        return x_disp, h_disp

    debug = {
        "Nfine": Nfine,
        "step": step,
        "base_idx": base_idx,
        "disp": disp,
        "disp_idx": disp_idx,
        "ks": ks,
        "pk": pk,
        "sigma": sigma,
    }
    return x_disp, h_disp, debug