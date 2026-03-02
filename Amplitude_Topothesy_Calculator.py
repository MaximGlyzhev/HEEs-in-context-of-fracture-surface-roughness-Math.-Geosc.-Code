import numpy as np
from fBm_circulant_embedding import circulant_fBm, fBm_on_unit_interval


# Evaluates range of a profile.

def array_range(arr):
    """
    Return the range (max - min) of a NumPy-compatible array.

    Parameters
    ----------
    arr : array-like
        Input array (list, tuple, or np.ndarray)

    Returns
    -------
    float
        max(arr) - min(arr)
    """
    a = np.asarray(arr)

    if a.size == 0:
        raise ValueError("Cannot compute range of an empty array")

    return np.max(a) - np.min(a)

# Rescales a profile.


def scale_to_range(arr, target_range):
    """
    Linearly scale values so that max(arr)-min(arr) == target_range.

    Parameters
    ----------
    arr : array-like (list, tuple, np.ndarray)
        Input values.
    target_range : float
        Desired range (max - min) after scaling.

    Returns
    -------
    np.ndarray
        Scaled array.
    """
    a = np.asarray(arr, dtype=float)

    if a.size == 0:
        return a  # empty array

    current_min = a.min()
    current_max = a.max()
    current_range = current_max - current_min

    if current_range == 0:
        # all values identical -> return zeros (range 0)
        return np.zeros_like(a)

    scale_factor = target_range / current_range
    return (a - current_min) * scale_factor

# Evaluates the average ampltiude of an fBm of length L with Hurst exponent H.


def amplitude_from_topothesy(L, H, topothesy,rep):
    Amplitudes = []
    for _ in range(rep):
       fBm = circulant_fBm(H, L, topothesy)[0]
       Amplitudes.append(array_range(fBm))
    mean_amplitude = np.mean(Amplitudes)
    return mean_amplitude

# Evaluates the average ampltiude of an fBm of length 1 with Hurst exponent H.
# The higher the point density is the more accurate is the estimate of the average amplitude.

def amplitude_from_topothesy_on_unit_interval(point_density, H, topothesy,rep):
    Amplitudes = []
    for _ in range(rep):
        fBm = fBm_on_unit_interval(H, point_density, topothesy)
        Amplitudes.append(array_range(fBm))
    mean_amplitude = np.mean(Amplitudes)
    return mean_amplitude

# Estimates the topothesy of an fBm profile of length L with Hurst exponent H
# The higher rep is the more accurate is the estimate of the topothesy.


def topothesy_from_amplitude(L,H,amplitude,rep):
    ranges_list = [0]
    for _ in range(rep):
        fBm = circulant_fBm(H, L)[0]
        ranges_list.append(array_range(fBm))
    average_range = np.mean(ranges_list)
    topothesy = amplitude/average_range
    return topothesy



    





    
        

       
     

   