import numpy as np
from typing import Optional, Dict, Any, Tuple
from White_Noise import add_white_noise
from fBm_with_displacements import fBm_with_displacements
from Profile_n_bit_conversion import quantize_profile_nbit

# L should be interpreted as profile length here

def fBm_with_simulated_measurement_errors(
    H: float,
    L: int,
    topothesy: float,
    M: int,
    p_nd: float,
    I: float,
    n_bits: int,
    rng: Optional[np.random.Generator] = None,
    return_debug: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pipeline:
      1) Generate fBm with x-axis displacement errors.
      2) Add white noise of intensity I to heights.
      3) Quantize heights to n_bits.

    Returns
    -------
    fbm_all_errors : np.ndarray
        Final profile after all simulated measurement errors.

    If return_debug=True:
        (fbm_all_errors, debug_dict)
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- basic validation (optional but recommended)
    if L < 2:
        raise ValueError("L must be >= 2.")
    if M < 1:
        raise ValueError("M must be >= 1.")
    if not (0.0 < p_nd < 1.0):
        raise ValueError("p_nd must be in (0,1).")
    if I < 0.0:
        raise ValueError("Noise intensity I must be >= 0.")
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1.")
    # If you want to strictly match the text:
    # if n_bits not in (8, 16, 32):
    #     raise ValueError("n_bits must be one of {8,16,32}.")

    # --- 1) displacement
    if return_debug:
        x_disp, h_disp, disp_debug = fBm_with_displacements(
            H=H, L=L, topothesy=topothesy, M=M, p_nd=p_nd, rng=rng, return_debug=True
        )
    else:
        x_disp, h_disp = fBm_with_displacements(
            H=H, L=L, topothesy=topothesy, M=M, p_nd=p_nd, rng=rng, return_debug=False
        )
        disp_debug = None

    # --- 2) add vertical white noise
    h_noisy = add_white_noise(h_disp, I=I, rng=rng)

    # --- 3) quantize to n bits (and keep original range)
    h_quant, (hmin, hmax) = quantize_profile_nbit(h_noisy, n_bits=n_bits)

    if not return_debug:
        return h_quant

    debug: Dict[str, Any] = {
        "x_disp": x_disp,
        "h_disp": h_disp,
        "h_noisy": h_noisy,
        "quant_range_before": (hmin, hmax),
        "n_bits": n_bits,
        "I": I,
        "M": M,
        "p_nd": p_nd,
    }
    if disp_debug is not None:
        debug["displacement_debug"] = disp_debug

    return h_quant, debug
