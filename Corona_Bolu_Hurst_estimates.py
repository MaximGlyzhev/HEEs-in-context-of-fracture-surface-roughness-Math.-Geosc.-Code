"""
Estimate Hurst exponents (H) for gridded fault-surface profiles in x- and y-direction
using FIVE methods (CP, PSD, DFA, SVA, HFD) provided by you.

Inputs:
  - /mnt/data/bolu2_grids.npz
  - /mnt/data/coronaA_grids.npz
  - /mnt/data/CP.py, PSD.py, DFA.py, SVA.py, HFD.py

What it does:
  1) For each dataset and each direction (x-profiles = rows, y-profiles = columns),
     compute H for every profile (skipping invalid/too-short profiles).
  2) Report the mean H (and std, n used) per method & direction.
  3) For user-chosen specific row/column indices, compare the 5 H estimates per profile
     (prints a table + optional bar plots).

Notes on NaNs:
  - Profiles may contain NaNs (especially Corona-A).
  - We take the LONGEST contiguous finite segment of each profile before estimating H.
    This avoids stitching across gaps.

CP method:
  - Uses step_size = 3 exactly as requested.
  
  
This file is supplementary material and is not required for any analysis of data generation 
procedure of the paper.
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt

# Make sure we can import your method files from /mnt/data
# Path to the folder with npz files
sys.path.insert(0, "/mnt/data")

from CP import estimate_change_probability, hurst_from_change_probability
from PSD import hurst_psd_welch_text
from DFA import dfa
from SVA import sva
from HFD import higuchi_fd



# If the npz files are renamed, the new file names should be inserted here
NPZ_FILES = {
    "Bolu-2": "bolu2_grids.npz",
    "Corona-A": "coronaA_grids.npz",
}

USE_DETRENDED = True        # True -> Z_detrended; False -> Z_grid
CP_STEP_SIZE = 10            # required
MIN_VALID_LEN = 50          # minimum contiguous finite samples needed to estimate H

# Choose specific profiles to compare across methods:
# - rows are x-direction profiles (constant y): Z[row, :]
# - cols are y-direction profiles (constant x): Z[:, col]
ROWS_TO_COMPARE = []   # edit as desired
COLS_TO_COMPARE = []   # edit as desired

MAKE_BAR_PLOTS_FOR_SELECTED = True


# -----------------------------
# Helpers
# -----------------------------
def longest_finite_segment(x: np.ndarray) -> np.ndarray:
    """
    Return the longest contiguous segment of finite values from a 1D array.
    If no finite values exist, returns an empty array.
    """
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.array([], dtype=float)

    idx = np.flatnonzero(finite)
    # find breaks where indices are not consecutive
    breaks = np.where(np.diff(idx) > 1)[0]

    # segments in idx-space
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(idx) - 1]

    # pick the longest
    lengths = (ends - starts + 1)
    k = int(np.argmax(lengths))
    seg_idx = idx[starts[k] : ends[k] + 1]
    return x[seg_idx].astype(float)


def safe_hurst_estimates(profile_1d: np.ndarray) -> dict[str, float]:
    """
    Compute Hurst estimates for ONE profile using the 5 methods.
    Returns dict {method_name: H} for methods that succeed.
    Methods that fail return np.nan.
    """
    out: dict[str, float] = {}

    x = np.asarray(profile_1d, dtype=float).ravel()
    x = longest_finite_segment(x)

    if x.size < MIN_VALID_LEN:
        return {k: np.nan for k in ["CP", "PSD", "DFA", "SVA", "HFD"]}

    # --- CP ---
    try:
        p = estimate_change_probability(x, stepsize=CP_STEP_SIZE)
        out["CP"] = float(hurst_from_change_probability(p))
    except Exception:
        out["CP"] = np.nan

    # --- PSD (Welch) ---
    try:
        out["PSD"] = float(hurst_psd_welch_text(x))  # defaults from your file
    except Exception:
        out["PSD"] = np.nan

    # --- DFA ---
    try:
        H, _, _ = dfa(x,1, 10, 0.05, 10)  # not default values. Values were set to compare to SVA and HFD
        out["DFA"] = float(H)
    except Exception:
        out["DFA"] = np.nan

    # --- SVA ---
    try:
        H, _, _ = sva(x)  # defaults from your file
        out["SVA"] = float(H)
    except Exception:
        out["SVA"] = np.nan

    # --- HFD ---
    try:
        _, H = higuchi_fd(x)  # defaults from your file
        out["HFD"] = float(H)
    except Exception:
        out["HFD"] = np.nan

    return out


def summarize_direction(Z: np.ndarray, direction: str) -> dict[str, dict[str, float]]:
    """
    Compute H for all profiles in one direction and summarize by method.

    direction:
      - "x": profiles along x (rows): Z[j, :]
      - "y": profiles along y (cols): Z[:, i]

    Returns:
      summary[method] = {"mean":..., "std":..., "n":...}
    """
    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")

    methods = ["CP", "PSD", "DFA", "SVA", "HFD"]
    Hvals = {m: [] for m in methods}

    if direction == "x":
        iterable = (Z[j, :] for j in range(Z.shape[0]))
    else:
        iterable = (Z[:, i] for i in range(Z.shape[1]))

    for prof in iterable:
        est = safe_hurst_estimates(prof)
        for m in methods:
            if np.isfinite(est[m]):
                Hvals[m].append(est[m])

    summary: dict[str, dict[str, float]] = {}
    for m in methods:
        arr = np.array(Hvals[m], dtype=float)
        summary[m] = {
            "mean": float(np.nanmean(arr)) if arr.size else np.nan,
            "std": float(np.nanstd(arr)) if arr.size else np.nan,
            "n": int(arr.size),
        }
    return summary


def print_summary_table(dataset_name: str, direction: str, summary: dict[str, dict[str, float]]):
    print(f"\n{dataset_name} — direction {direction}")
    print("-" * (len(dataset_name) + 16))
    print(f"{'Method':<6}  {'mean(H)':>10}  {'std(H)':>10}  {'n':>6}")
    for m in ["CP", "PSD", "DFA", "SVA", "HFD"]:
        s = summary[m]
        print(f"{m:<6}  {s['mean']:>10.4f}  {s['std']:>10.4f}  {s['n']:>6d}")


def compare_selected_profiles(
    Z: np.ndarray,
    dataset_name: str,
    *,
    rows: list[int],
    cols: list[int],
    make_bar_plots: bool = True,
):
    """
    Compare the 5 methods for selected row/col indices.
    Prints per-profile H estimates; optionally makes bar plots.
    """
    methods = ["CP", "PSD", "DFA", "SVA", "HFD"]
    ny, nx = Z.shape

    # --- rows (x-profiles) ---
    for r in rows:
        if not (0 <= r < ny):
            continue
        est = safe_hurst_estimates(Z[r, :])
        print(f"\n{dataset_name} — Row {r} (x-profile)")
        print(f"{'Method':<6}  {'H':>10}")
        for m in methods:
            print(f"{m:<6}  {est[m]:>10.4f}")

        if make_bar_plots:
            vals = [est[m] for m in methods]
            plt.figure(figsize=(6, 4))
            plt.bar(methods, vals)
            plt.ylim(0, 1)
            plt.ylabel("H")
            plt.title(f"{dataset_name}: H estimates (row {r}, x-profile)")
            plt.tight_layout()
            plt.show()

    # --- cols (y-profiles) ---
    for c in cols:
        if not (0 <= c < nx):
            continue
        est = safe_hurst_estimates(Z[:, c])
        print(f"\n{dataset_name} — Col {c} (y-profile)")
        print(f"{'Method':<6}  {'H':>10}")
        for m in methods:
            print(f"{m:<6}  {est[m]:>10.4f}")

        if make_bar_plots:
            vals = [est[m] for m in methods]
            plt.figure(figsize=(6, 4))
            plt.bar(methods, vals)
            plt.ylim(0, 1)
            plt.ylabel("H")
            plt.title(f"{dataset_name}: H estimates (col {c}, y-profile)")
            plt.tight_layout()
            plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    for name, path in NPZ_FILES.items():
        d = np.load(path)
        Z = d["Z_detrended"] if USE_DETRENDED else d["Z_grid"]

        print(f"\n=== {name} ===")
        print(f"Using surface: {'Z_detrended' if USE_DETRENDED else 'Z_grid'}")
        print(f"Grid shape (ny, nx): {Z.shape}")

        # Average H in x direction (rows)
        sx = summarize_direction(Z, "x")
        print_summary_table(name, "x (rows: x-profiles)", sx)

        # Average H in y direction (cols)
        sy = summarize_direction(Z, "y")
        print_summary_table(name, "y (cols: y-profiles)", sy)

        # Compare selected profiles
        compare_selected_profiles(
            Z,
            name,
            rows=ROWS_TO_COMPARE,
            cols=COLS_TO_COMPARE,
            make_bar_plots=MAKE_BAR_PLOTS_FOR_SELECTED,
        )


if __name__ == "__main__":
    main()