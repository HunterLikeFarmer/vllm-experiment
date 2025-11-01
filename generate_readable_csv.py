#!/usr/bin/env python3
# Usage:
#   python generate_readable_csv.py results.csv normalized.csv

import csv, sys, argparse

COLUMNS_V1 = [
    "timestamp","model","host","port","tp","gpu_mem",
    "max_num_seqs","max_num_batched_tokens","block_size",
    "mode_chunked_prefill","mode_prefix_caching","disable_custom_all_reduce",
    "concurrency","num_requests","max_new_tokens","temperature",
    "success_count","error_count","wall_time_s",
    "total_prompt_tokens","total_completion_tokens","total_tokens",
    "throughput_tokens_per_s","goodput_tokens_per_s","reqs_per_s","notes"
]

# Matches your current run_vllm_sweep.py header (TTFT/TBT percentiles + log_path)
COLUMNS_V2 = COLUMNS_V1 + [
    "ttft_ms_p50","ttft_ms_p95","ttft_ms_p99",
    "tbt_ms_p50","tbt_ms_p95","tbt_ms_p99",
    "log_path"
]

def looks_like_header(row):
    lc = [c.strip().lower() for c in row]
    # consider it a header if >= 5 known names are present
    return sum(1 for name in COLUMNS_V2 if name in lc) >= 5 or \
           sum(1 for name in COLUMNS_V1 if name in lc) >= 5

def choose_header(width, forced=None):
    if forced == "v1":
        return COLUMNS_V1
    if forced == "v2":
        return COLUMNS_V2
    # Auto-detect by width
    if width == len(COLUMNS_V2):
        return COLUMNS_V2
    if width == len(COLUMNS_V1):
        return COLUMNS_V1
    # Prefer v2 (superset); we'll pad/truncate as needed
    return COLUMNS_V2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--force", choices=["v1","v2"], default=None,
                    help="Force a specific header shape; otherwise auto-detect.")
    args = ap.parse_args()

    with open(args.inp, newline="") as fin:
        rows = list(csv.reader(fin))

    if not rows:
        with open(args.out, "w", newline="") as fout:
            csv.writer(fout).writerow(COLUMNS_V2)
        print(f"Wrote empty {args.out} with v2 header.")
        return

    # Strip existing header if present
    data_rows = rows[1:] if looks_like_header(rows[0]) else rows

    # Find widest row to guide detection
    width = max((len(r) for r in data_rows), default=0)
    header = choose_header(width, forced=args.force)

    # Write normalized file
    with open(args.out, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header)
        target_len = len(header)
        for row in data_rows:
            row = (row + [""] * target_len)[:target_len]
            w.writerow(row)

    print(f"Wrote {args.out} with {len(data_rows)} rows and header: {('v2' if header==COLUMNS_V2 else 'v1')}")

if __name__ == "__main__":
    main()
