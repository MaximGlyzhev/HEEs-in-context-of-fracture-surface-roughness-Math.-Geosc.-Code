from __future__ import annotations

import numpy as np
import pandas as pd
import os

from Berry_Esseen_bounds import universal_berry_essen_bound


"""
This code evaluates universal BEBs for the method comparison plot.
PaperStats.csv contains the method statistics necessary to evaluate universal BEBs.
The path to the PaperStats.csv file should be inserted in the main function.

"""

# -----------------------------
# Your helpers (same as yours)
# -----------------------------
def load_paperstats(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .parquet")

    required = {"H_true", "L", "method", "std", "atm3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["H_true"] = df["H_true"].astype(float)
    df["L"] = df["L"].astype(int)
    df["method"] = df["method"].astype(str)
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    df["atm3"] = pd.to_numeric(df["atm3"], errors="coerce")
    return df


def build_sigma_rho(
    df: pd.DataFrame,
    *,
    method: str,
    L: int,
    H_values=None,
    drop_nonfinite: bool = True,
):
    sub = df[(df["method"] == method) & (df["L"] == int(L))].copy()
    if sub.empty:
        available = df[["method", "L"]].drop_duplicates().sort_values(["method", "L"])
        raise ValueError(
            f"No rows found for method={method!r}, L={L}. "
            f"Available combos:\n{available.to_string(index=False)}"
        )

    if H_values is not None:
        Hv = np.array(list(H_values), dtype=float)
        sub = sub[sub["H_true"].isin(Hv)]
        if sub.empty:
            raise ValueError(f"No rows left after filtering to H_values={list(H_values)}")

    sub = (
        sub.groupby("H_true", as_index=False)[["std", "atm3"]]
        .mean(numeric_only=True)
        .sort_values("H_true")
        .reset_index(drop=True)
    )

    if drop_nonfinite:
        mask = np.isfinite(sub["std"].to_numpy()) & np.isfinite(sub["atm3"].to_numpy())
        sub = sub.loc[mask].reset_index(drop=True)

    S = sub["H_true"].to_numpy(dtype=float)
    sigma = sub["std"].to_numpy(dtype=float)
    rho = sub["atm3"].to_numpy(dtype=float)

    if S.size == 0:
        raise ValueError("After filtering, no finite (std, atm3) values remain.")

    sigma_by_H = {float(H): float(s) for H, s in zip(S, sigma)}
    rho_by_H = {float(H): float(r) for H, r in zip(S, rho)}
    return S, sigma_by_H, rho_by_H


# -----------------------------
# Compute + print extra_data
# -----------------------------
def main():
    # EDIT THIS if needed
    stats_path = ""  # <-- in this environment, your file is here

    methods = ["CP", "DFA", "HFD", "PSD", "SVA"]
    L_values = [1000, 2000, 5000]

    # This is the S-grid you described
    S_grid = np.round(np.arange(0.1, 1.0, 0.1), 1)  # 0.1 ... 0.9

    # These correspond to your plot titles:
    # bottom_titles = [
    #   (p,eps)=(0.10,0.05),
    #   (p,eps)=(0.10,0.005),
    #   (p,eps)=(0.05,0.005)
    # ]
    # NOTE: Your plotting code uses: extra_data = [extra_data3, extra_data1, extra_data2]
    cases = [
        ("extra_data3", 0.10, 0.05),
        ("extra_data1", 0.10, 0.005),
        ("extra_data2", 0.05, 0.005),
    ]

    df = load_paperstats(stats_path)

    results = {}
    for name, p, eps in cases:
        arr = np.zeros((len(methods), len(L_values)), dtype=float)

        for i, method in enumerate(methods):
            for j, L in enumerate(L_values):
                S, sigma_by_H, rho_by_H = build_sigma_rho(
                    df,
                    method=method,
                    L=L,
                    H_values=S_grid,     # restrict to 0.1..0.9
                )

                n_univ, H_worst = universal_berry_essen_bound(
                    epsilon=eps,
                    p=p,
                    S=S,
                    sigma_by_H=sigma_by_H,
                    rho_by_H=rho_by_H,
                )

                # Your description: "bounds multiplied by profile lengths"
                arr[i, j] = float(n_univ) * float(L)

        results[name] = arr

    # Pretty-print as pasteable numpy arrays
    def fmt_array(A: np.ndarray) -> str:
        # print as ints if they are essentially integers
        if np.all(np.isfinite(A)) and np.all(np.abs(A - np.round(A)) < 1e-9):
            A = np.round(A).astype(int)
            rows = ["[" + ", ".join(str(x) for x in row) + "]" for row in A]
        else:
            rows = ["[" + ", ".join(f"{x:.6g}" for x in row) + "]" for row in A]
        return "np.array([\n        " + ",\n        ".join(rows) + "\n    ])"

    for name, _, _ in cases:
        print(f"{name} = {fmt_array(results[name])}\n")

    print("extra_data = [extra_data3, extra_data1, extra_data2]")


if __name__ == "__main__":
    main()
