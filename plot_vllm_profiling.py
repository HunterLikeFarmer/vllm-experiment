#!/usr/bin/env python3
"""
Plot ttft_ms_p95 and tbt_ms_p99 vs row index (sorted by ttft_ms_p99),
skipping rows that aren't "ready".
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_ready_mask(df: pd.DataFrame, ready_col: str | None, ready_values: list[str] | None):
    """
    Returns (mask, used_ready_col). If no ready column is found, mask is all True.
    """
    # Auto-detect a readiness column if not provided
    if ready_col is None:
        for c in df.columns:
            if c.strip().lower() in {"ready", "is_ready", "status", "state", "completed", "done"}:
                ready_col = c
                break

    if ready_col is None or ready_col not in df.columns:
        return pd.Series(True, index=df.index), None

    col = df[ready_col]
    if pd.api.types.is_bool_dtype(col):
        return col.astype(bool), ready_col

    # Treat these string/number encodings as "ready"
    allowed = set(v.strip().lower() for v in (ready_values or
             ["true", "t", "1", "yes", "y", "ready", "done", "complete", "completed"]))
    col_norm = col.astype(str).str.strip().str.lower()
    return col_norm.isin(allowed), ready_col


def main():
    ap = argparse.ArgumentParser(description="Scatter plot of ttft_ms_p95 and tbt_ms_p99 vs row index sorted by ttft_ms_p99.")
    ap.add_argument("--csv", default="results.csv", help="Path to input CSV (default: results.csv)")
    ap.add_argument("--out", default="ttft_tbt_scatter.png", help="Output image path (default: ttft_tbt_scatter.png)")
    ap.add_argument("--ready-col", default=None, help="Name of the column indicating readiness (optional)")
    ap.add_argument("--ready-values", nargs="*", default=None,
                    help="Values considered 'ready' in the ready column (e.g., true yes ready 1)")
    ap.add_argument("--show", action="store_true", help="Show an interactive window")
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]

    required = ["ttft_ms_p99", "ttft_ms_p95", "tbt_ms_p99"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

    # Coerce required metrics to numeric and keep only finite rows
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    numeric_ok = df[required].notna().all(axis=1) & np.isfinite(df[required]).all(axis=1)

    # Readiness filter
    ready_mask, used_col = build_ready_mask(df, args.ready_col, args.ready_values)

    # Filter + sort
    keep = numeric_ok & ready_mask
    df_f = df.loc[keep, required].copy().sort_values("ttft_ms_p99", ascending=True).reset_index(drop=True)

    if df_f.empty:
        raise SystemExit("No rows to plot after filtering for readiness and numeric values.")

    # X-axis: row order after sorting (1..N)
    x = np.arange(1, len(df_f) + 1)

    # Plot (single chart, two series)
    plt.figure(figsize=(10, 5))
    plt.scatter(x, df_f["ttft_ms_p95"], label="ttft_ms_p95")
    plt.scatter(x, df_f["tbt_ms_p99"], label="tbt_ms_p99")
    plt.xlabel("Row (sorted by ttft_ms_p99)")
    plt.ylabel("Milliseconds")
    plt.title("ttft_ms_p95 and tbt_ms_p99 vs. Row Index (sorted by ttft_ms_p99)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    if args.show:
        plt.show()

    used_str = used_col if used_col else "none (only numeric validity used)"
    print(f"Plotted {len(df_f)} rows out of {len(df)} total.")
    print(f"Readiness column used: {used_str}")
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
