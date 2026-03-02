import numpy as np
import matplotlib.pyplot as plt
from fBm_circulant_embedding import circulant_fBm

"""
This code creates a histogram of amplitudes for independent fBm realisation.
This code is not required for any analysis or data generation procedure of the study.
However, the fBm histograms can be compared to histogram of fracture surface data sets
 like Bolu-2 or Corona-A.
"""

# ------------------------------------------------------------
# Script: generate 563 profiles (each length 701) and histogram
# ------------------------------------------------------------
def main():
    # Parameters you can adjust
    H = 0.4
    m = 701
    n_profiles = 563
    topothesy = 1.0
    seed = 12345
    bins = 30

    rng = np.random.default_rng(seed)

    # Generate profiles and compute peak-to-peak amplitudes
    amplitudes = np.empty(n_profiles, dtype=float)

    for i in range(n_profiles):
        fbm, _ = circulant_fBm(H=H, m=m, topothesy=topothesy, rng=rng)
        amplitudes[i] = fbm.max() - fbm.min()

    print("Generated profiles:", n_profiles)
    print("Points per profile:", m)
    print("H:", H, "topothesy:", topothesy)
    print("Mean amplitude:", amplitudes.mean())
    print("Std amplitude:", amplitudes.std())
    print("Min/Max amplitude:", amplitudes.min(), amplitudes.max())

    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(amplitudes, bins=bins)
    plt.xlabel("Peak-to-peak amplitude [same units as topothesy]")
    plt.ylabel("Count")
    plt.title(f"fBm amplitude histogram (N={n_profiles}, m={m}, H={H})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()