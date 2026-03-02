import numpy as np
import random


def add_white_noise(profile: np.ndarray, I: float,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add white Gaussian noise of intensity I to a profile.

    Parameters
    ----------
    profile : np.ndarray
        Input signal/profile.
    I : float
        Noise intensity = standard deviation of the white noise.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    noisy_profile : np.ndarray
        Profile + white noise (same shape as input).
    """
    if I < 0:
        raise ValueError("Noise intensity must be nonnegative.")

    if rng is None:
        rng = np.random.default_rng()
    variance = I*I
    noise = rng.normal(loc=0.0, scale=variance, size=profile.shape)
    return profile + noise




### Uniform noise was not used in the study
def uniform_random(a, b):
    """
    Returns a uniformly distributed random float in the interval [a, b].
    """
    return random.uniform(a, b)

def add_uniform_noise(profile: np.ndarray, I: float,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add white uniform noise of intensity I (std dev) to a profile.

    Parameters
    ----------
    profile : np.ndarray
        Input signal/profile.
    I : float
        Noise intensity = standard deviation of the uniform noise.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    noisy_profile : np.ndarray
        Profile + uniform white noise (same shape as input).
    """
    if I < 0:
        raise ValueError("Noise intensity must be nonnegative.")

    if rng is None:
        rng = np.random.default_rng()

    # For uniform noise with std = I:
    # interval = [-sqrt(3)*I, sqrt(3)*I]
    half_width = np.sqrt(3.0) * I

    noise = rng.uniform(
        low=-half_width,
        high=half_width,
        size=profile.shape
    )

    return profile + noise