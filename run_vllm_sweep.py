#!/usr/bin/env python3
"""
run_vllm_sweep.py
Orchestrates a sweep of vLLM server parameters and records throughput/goodput (+ TTFT/TBT if client streams).

Usage (example):
  python run_vllm_sweep.py \
    --model Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 8000 \
    --gpu-mem 0.80 \
    --tensor-parallel 1 \
    --max-num-seqs 64 128 256 \
    --max-num-batched-tokens 1024 2048 4096 \
    --block-size 16 32 \
    --modes chunked prefix \
    --concurrency 16 --num-requests 200 --max-new-tokens 128 \
    --prompts-file prompts.txt \
    --client-stream \
    --out results.csv
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    # Allow single or multiple values for GPU mem & other knobs
    p.add_argument("--gpu-mem", type=float, nargs="+", default=[0.8],
                   help="One or more values for --gpu-memory-utilization (e.g., 0.75 0.8 0.9)")
    p.add_argument("--tensor-parallel", type=int, nargs="+", default=[1])
    p.add_argument("--max-num-seqs", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--max-num-batched-tokens", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--block-size", type=int, nargs="+", default=[16, 32])

    # Mode flags
    p.add_argument("--modes", nargs="+", default=["chunked", "prefix"],
                   choices=["chunked", "prefix"],
                   help="chunked -> --enable-chunked-prefill, prefix -> --enable-prefix-caching")

    # Important: default False so the flag actually toggles ON when passed
    p.add_argument("--disable-custom-all-reduce", action="store_true", default=False)

    # Load/client shape
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompts-file", type=str, default=None)

    # Output / plumbing
    p.add_argument("--out", type=str, default="results.csv")
    p.add_argument("--logdir", type=str, default="logs")
    p.add_argument("--timeout-ready", type=int, default=900, help="Seconds to wait for server readiness per run.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python launcher to run llm_load_client.py")

    # Ask client to stream so TTFT/TBT are measurable. Optional.
    p.add_argument("--client-stream", action="store_true",
                   help="Pass --stream to llm_load_client.py to measure TTFT/TBT.")
    return p.parse_args()


def poll_ready(url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            req = Request(url, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (URLError, HTTPError) as e:
            last_err = e
        time.sleep(2)
    if last_err:
        print(f"[warn] Last readiness error: {last_err}", file=sys.stderr)
    return False


def start_server(cmd, log_path):
    lf = open(log_path, "w")
    print(f"[info] Launching: {' '.join(shlex.quote(x) for x in cmd)}")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return proc, lf


def stop_server(proc, lf):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    lf.close()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.logdir)
    out_path = Path(args.out)

    # Write header if file doesn't exist
    if not out_path.exists():
        with open(out_path, "w") as f:
            f.write(",".join([
                "timestamp","model","host","port","tp","gpu_mem",
                "max_num_seqs","max_num_batched_tokens","block_size",
                "mode_chunked_prefill","mode_prefix_caching","disable_custom_all_reduce",
                "concurrency","num_requests","max_new_tokens","temperature",
                "success_count","error_count","wall_time_s",
                "total_prompt_tokens","total_completion_tokens","total_tokens",
                "throughput_tokens_per_s","goodput_tokens_per_s","reqs_per_s","notes",
                "ttft_ms_p50","ttft_ms_p95","ttft_ms_p99",
                "tbt_ms_p50","tbt_ms_p95","tbt_ms_p99",
                "log_path"
            ]) + "\n")

    # Build combos
    combos = []
    for tp in args.tensor_parallel:
        for gpu_mem in args.gpu_mem:
            for mns in args.max_num_seqs:
                for mnbt in args.max_num_batched_tokens:
                    if mnbt < mns:  # hard constraint
                        continue
                    for bs in args.block_size:
                        for mode in args.modes:
                            combos.append((tp, gpu_mem, mns, mnbt, bs, mode))

    if not combos:
        print("[error] No valid parameter combinations after constraints.", file=sys.stderr)
        sys.exit(2)

    for idx, (tp, gpu_mem, mns, mnbt, bs, mode) in enumerate(combos, start=1):
        ts = int(time.time())
        tag = f"tp{tp}_mem{gpu_mem}_mns{mns}_mnbt{mnbt}_bs{bs}_{mode}"
        log_path = os.path.join(args.logdir, f"serve_{tag}.log")

        cmd = [
            "vllm", "serve", args.model,
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(gpu_mem),
            "--max-num-seqs", str(mns),
            "--max-num-batched-tokens", str(mnbt),
            "--block-size", str(bs),
            "--host", args.host,
            "--port", str(args.port),
        ]
        if args.disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")

        # Mutually exclusive modes in this script version
        mode_chunked = (mode == "chunked")
        mode_prefix = (mode == "prefix")
        if mode_chunked:
            cmd.append("--enable-chunked-prefill")
        if mode_prefix:
            cmd.append("--enable-prefix-caching")

        # Start server
        proc, lf = start_server(cmd, log_path)

        # Wait for readiness on /v1/models
        ready = poll_ready(f"http://{args.host}:{args.port}/v1/models", timeout_s=args.timeout_ready)
        if not ready:
            print(f"[error] Server not ready for combo {tag}. Skipping. See log: {log_path}", file=sys.stderr)
            stop_server(proc, lf)
            with open(out_path, "a") as f:
                f.write(",".join(map(str, [
                    ts,args.model,args.host,args.port,tp,gpu_mem,
                    mns,mnbt,bs,
                    int(mode_chunked),int(mode_prefix),int(args.disable_custom_all_reduce),
                    args.concurrency,args.num_requests,args.max_new_tokens,args.temperature,
                    0,1,0.0,0,0,0,0.0,0.0,0.0,"not_ready",
                    0.0,0.0,0.0, 0.0,0.0,0.0,
                    log_path
                ])) + "\n")
            continue

        # Run client
        client_cmd = [
            args.python, "llm_load_client.py",
            "--server-url", f"http://{args.host}:{args.port}/v1",
            "--model", args.model,
            "--concurrency", str(args.concurrency),
            "--num-requests", str(args.num_requests),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
        ]
        if args.prompts_file:
            client_cmd += ["--prompts-file", args.prompts_file]
        if args.client_stream:
            client_cmd += ["--stream"]

        print(f"[info] Running client for combo {tag} ...")
        t0 = time.time()
        try:
            client_out = subprocess.check_output(client_cmd, stderr=subprocess.STDOUT, timeout=3600)
            _ = time.time() - t0
            try:
                metrics = json.loads(client_out.decode("utf-8"))
            except json.JSONDecodeError:
                print("[error] Failed to parse client JSON. See stdout below:", file=sys.stderr)
                print(client_out.decode("utf-8"), file=sys.stderr)
                metrics = {
                    "success_count": 0, "error_count": args.num_requests,
                    "wall_time_s": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "throughput_tokens_per_s": 0.0,
                    "goodput_tokens_per_s": 0.0,
                    "reqs_per_s": 0.0,
                    "notes": "client_json_parse_error",
                    "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0, "ttft_ms_p99": 0.0,
                    "tbt_ms_p50": 0.0, "tbt_ms_p95": 0.0, "tbt_ms_p99": 0.0
                }
        except subprocess.TimeoutExpired:
            metrics = {
                "success_count": 0, "error_count": args.num_requests,
                "wall_time_s": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "throughput_tokens_per_s": 0.0,
                "goodput_tokens_per_s": 0.0,
                "reqs_per_s": 0.0,
                "notes": "client_timeout",
                "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0, "ttft_ms_p99": 0.0,
                "tbt_ms_p50": 0.0, "tbt_ms_p95": 0.0, "tbt_ms_p99": 0.0
            }
        finally:
            stop_server(proc, lf)

        # Append row
        with open(out_path, "a") as f:
            f.write(",".join(map(str, [
                ts,args.model,args.host,args.port,tp,gpu_mem,
                mns,mnbt,bs,
                int(mode_chunked),int(mode_prefix),int(args.disable_custom_all_reduce),
                args.concurrency,args.num_requests,args.max_new_tokens,args.temperature,
                metrics.get("success_count",0),metrics.get("error_count",0),metrics.get("wall_time_s",0.0),
                metrics.get("total_prompt_tokens",0),metrics.get("total_completion_tokens",0),metrics.get("total_tokens",0),
                metrics.get("throughput_tokens_per_s",0.0),metrics.get("goodput_tokens_per_s",0.0),metrics.get("reqs_per_s",0.0),
                metrics.get("notes",""),
                metrics.get("ttft_ms_p50",0.0),metrics.get("ttft_ms_p95",0.0),metrics.get("ttft_ms_p99",0.0),
                metrics.get("tbt_ms_p50",0.0),metrics.get("tbt_ms_p95",0.0),metrics.get("tbt_ms_p99",0.0),
                log_path
            ])) + "\n")

        print(f"[info] Done combo {idx}/{len(combos)} -> {tag}")

    print(f"[info] Sweep complete. Results in {out_path.resolve()}")


if __name__ == "__main__":
    main()
