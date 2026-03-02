import numpy as np
import matplotlib.pyplot as plt


"""
This is a supplementary code which is not used for any analysis or data generation
procedure of the study.
This code creates a histogram of amplitudes of the Bolu-2 and Corona-A data set.

"""




def profile_amplitudes(Z: np.ndarray):
    """
    Compute peak-to-peak amplitude (max-min) for each profile:
      - x-direction profiles = rows (constant y)
      - y-direction profiles = columns (constant x)
    NaNs are ignored.
    """
    amp_x = []
    for j in range(Z.shape[0]):
        prof = Z[j, :]
        prof = prof[np.isfinite(prof)]
        if prof.size > 0:
            amp_x.append(prof.max() - prof.min())
    amp_x = np.asarray(amp_x, dtype=float)

    amp_y = []
    for i in range(Z.shape[1]):
        prof = Z[:, i]
        prof = prof[np.isfinite(prof)]
        if prof.size > 0:
            amp_y.append(prof.max() - prof.min())
    amp_y = np.asarray(amp_y, dtype=float)

    return amp_x, amp_y


def analyze_dataset(npz_path: str, use_detrended: bool):
    """
    Loads one dataset and returns:
      - Lx, Ly
      - amp_x, amp_y
      - mean amplitudes
    """
    data = np.load(npz_path)

    xg = data["xg"]
    yg = data["yg"]
    dx = float(data["dx"])
    dy = float(data["dy"])

    Z = data["Z_detrended"] if use_detrended else data["Z_grid"]

    # profile lengths
    Lx = (len(xg) - 1) * dx
    Ly = (len(yg) - 1) * dy

    # amplitudes
    amp_x, amp_y = profile_amplitudes(Z)

    return {
        "Lx": Lx,
        "Ly": Ly,
        "amp_x": amp_x,
        "amp_y": amp_y,
        "mean_amp_x": float(np.mean(amp_x)) if amp_x.size else np.nan,
        "mean_amp_y": float(np.mean(amp_y)) if amp_y.size else np.nan,
        "dx": dx,
        "dy": dy,
        "shape": Z.shape,
        "nan_frac": float(np.mean(~np.isfinite(Z))),
    }


def plot_histograms(name: str, amp_x: np.ndarray, amp_y: np.ndarray, use_detrended: bool):
    """Improved histogram plotting (Fix A)."""
    label_surface = "detrended" if use_detrended else "raw"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    bins = 25  # fewer bins = cleaner ticks

    # ---- X direction histogram ----
    axes[0].hist(amp_x, bins=bins)
    axes[0].set_title(f"{name}: amplitude histogram (x profiles)")
    axes[0].set_xlabel("Amplitude [m]")
    axes[0].set_ylabel("Count")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    axes[0].tick_params(axis="x", labelrotation=30)

    # ---- Y direction histogram ----
    axes[1].hist(amp_y, bins=bins)
    axes[1].set_title(f"{name}: amplitude histogram (y profiles)")
    axes[1].set_xlabel("Amplitude [m]")
    axes[1].set_ylabel("Count")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    axes[1].tick_params(axis="x", labelrotation=30)

    fig.suptitle(f"{name} — {label_surface}")
    plt.show()



"""
The npz files should be in the same folder as the code. Alternaitvely, the full file path
can be used.
"""
def main():
    # ----------------------------
    # Inputs
    # ----------------------------
    files = {
        "Bolu-2": "bolu2_grids.npz",
        "Corona-A": "coronaA_grids.npz",
    }

    USE_DETRENDED = True  # set False for raw surface

    # ----------------------------
    # Analyze + plot
    # ----------------------------
    for name, path in files.items():
        res = analyze_dataset(path, use_detrended=USE_DETRENDED)

        print(f"\n{name} ({'detrended' if USE_DETRENDED else 'raw'})")
        print("-" * (len(name) + 15))
        print(f"Grid shape (ny, nx): {res['shape']}")
        print(f"dx, dy: {res['dx']:.6g}, {res['dy']:.6g} m")
        print(f"Profile length Lx: {res['Lx']:.6g} m")
        print(f"Profile length Ly: {res['Ly']:.6g} m")
        print(f"Mean amplitude (x profiles): {res['mean_amp_x']:.6g} m")
        print(f"Mean amplitude (y profiles): {res['mean_amp_y']:.6g} m")
        print(f"NaN fraction in Z: {res['nan_frac']:.3%}")

        plot_histograms(name, res["amp_x"], res["amp_y"], use_detrended=USE_DETRENDED)


if __name__ == "__main__":
    main()