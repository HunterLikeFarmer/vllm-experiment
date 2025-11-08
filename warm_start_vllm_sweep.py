#!/usr/bin/env python3
"""
run_vllm_sweep.py  (warm-start version)

Orchestrates a sweep of vLLM server parameters and records throughput/goodput (+ TTFT/TBT if client streams).

Adds detailed timestamp logging to each run's log file, including:
- combo start/end
- server launch start
- server ready (elapsed)
- client run start/end (elapsed)
- server stop start/done (elapsed)
- optional first/last request timestamps if provided by llm_load_client.py JSON

This version:
- Uses a WARM START: only restarts vLLM when any *server-side* knobs change.
- Always keeps DP=1 (vLLM default), sets PP automatically so TP*PP == #visible GPUs when possible.
- If not divisible, PP=1 and a warning is noted (partial GPU utilization).
- Logs both intended plan (TP,PP,DP) and the plan detected from vLLM logs (parsed).
- Produces one CSV row per client run (even if the server stays warm across multiple runs).

Server-side knobs we group by (trigger restart):
  model, host, port,
  tp, gpu_mem, max_num_seqs, max_num_batched_tokens, block_size, mode (chunked|prefix)

Client-side knobs (can vary with same server):
  concurrency, num_requests, max_new_tokens, temperature, prompts_file, stream

Usage (example):
  python run_vllm_sweep.py \
    --model Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 8000 \
    --gpu-mem 0.80 0.90 \
    --tensor-parallel 1 2 4 \
    --max-num-seqs 64 128 256 512 1024 2048 4096 8192 \
    --max-num-batched-tokens 64 128 256 512 1024 2048 4096 8192 \
    --block-size 16 32 \
    --modes chunked prefix \
    --concurrency 32 --num-requests 200 --max-new-tokens 128 \
    --prompts-file prompts.txt \
    --timeout-ready 180 \
    --client-stream \
    --out results.csv
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# ----------------- CLI -----------------

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

    # Mode flags (SERVER-SIDE)
    p.add_argument("--modes", nargs="+", default=["chunked", "prefix"],
                   choices=["chunked", "prefix"],
                   help="chunked -> --enable-chunked-prefill, prefix -> --enable-prefix-caching")

    # Important: default False so the flag toggles ON when passed
    p.add_argument("--disable-custom-all-reduce", action="store_true", default=False)

    # Client load shape (CLIENT-SIDE)
    p.add_argument("--concurrency", type=int, nargs="+", default=[16])
    p.add_argument("--num-requests", type=int, nargs="+", default=[200])
    p.add_argument("--max-new-tokens", type=int, nargs="+", default=[128])
    p.add_argument("--temperature", type=float, nargs="+", default=[0.0])
    p.add_argument("--prompts-file", type=str, nargs="+", default=[None])

    # Output / plumbing
    p.add_argument("--out", type=str, default="results_warm_start.csv")
    p.add_argument("--logdir", type=str, default="logs_warm")
    p.add_argument("--timeout-ready", type=int, default=900, help="Seconds to wait for server readiness per group.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python launcher to run llm_load_client.py")

    # Ask client to stream so TTFT/TBT are measurable. Optional.
    p.add_argument("--client-stream", action="store_true",
                   help="Pass --stream to llm_load_client.py to measure TTFT/TBT.")
    return p.parse_args()


# ----------------- Helpers -----------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def now_epoch() -> float:
    return time.time()


def to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def log_ts(lf, label: str, ts: float = None, extra: str = ""):
    if ts is None:
        ts = now_epoch()
    line = f"[timestamp] {label} | epoch={ts:.6f} | iso={to_iso(ts)}"
    if extra:
        line += f" | {extra}"
    lf.write(line + "\n")
    lf.flush()


def count_visible_gpus() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip() != ""])
    return 1


def compute_pp_for_all_gpus(tp: int, ngpus: int) -> Tuple[int, str]:
    """
    We keep DP=1, so we try to set PP so that tp*pp == ngpus.
    If not divisible, we fall back to pp=1 and warn.
    """
    if tp <= 0 or ngpus <= 0:
        return 1, f"invalid(tp={tp},ngpus={ngpus})"
    if ngpus % tp != 0:
        return 1, f"not_divisible(ngpus={ngpus},tp={tp})->pp=1,unused_gpus={ngpus - tp*1}"
    return max(1, ngpus // tp), ""


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


def start_server(cmd, log_path, env=None):
    lf = open(log_path, "w")
    print(f"[info] Launching: {' '.join(shlex.quote(x) for x in cmd)}")
    if env:
        for k in ["CUDA_VISIBLE_DEVICES"]:
            if k in env:
                lf.write(f"[env] {k}={env[k]}\n")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
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
    try:
        lf.flush()
    except Exception:
        pass
    lf.close()


def log_optional_client_marks(lf, metrics: dict):
    keys = [
        "first_request_sent_ts",
        "first_request_done_ts",
        "last_request_sent_ts",
        "last_request_done_ts",
    ]
    present = [k for k in keys if isinstance(metrics.get(k), (int, float)) and metrics.get(k) > 0]
    if not present:
        lf.write("[timestamp] client_request_marks | unavailable (client did not emit per-request timestamps)\n")
        lf.flush()
        return

    frs = metrics.get("first_request_sent_ts")
    frd = metrics.get("first_request_done_ts")
    lrs = metrics.get("last_request_sent_ts")
    lrd = metrics.get("last_request_done_ts")

    if frs:
        log_ts(lf, "first_request_sent", frs)
    if frd:
        log_ts(lf, "first_request_done", frd,
               extra=f"first_req_latency_s={frd - frs:.6f}" if (frs and frd and frd >= frs) else "")
    if lrs:
        log_ts(lf, "last_request_sent", lrs)
    if lrd:
        extra = []
        if frs and lrd and lrd >= frs:
            extra.append(f"window_first_send_to_last_done_s={lrd - frs:.6f}")
        if lrs and lrd and lrd >= lrs:
            extra.append(f"tail_latency_last_req_s={lrd - lrs:.6f}")
        log_ts(lf, "last_request_done", lrd, extra=" ".join(extra))


# -------- Parallel plan parsing --------

_PLAN_REGEXES = [
    re.compile(
        r"tensor_parallel_size=(\d+),\s*pipeline_parallel_size=(\d+),\s*data_parallel_size=(\d+)"
    ),
    re.compile(
        r"tensor_parallel_size'\s*:\s*(\d+).*?pipeline_parallel_size'\s*:\s*(\d+).*?data_parallel_size'\s*:\s*(\d+)",
        re.DOTALL
    ),
]

def parse_parallel_plan_from_text(text: str):
    for rgx in _PLAN_REGEXES:
        m = rgx.search(text)
        if m:
            tp_used = int(m.group(1))
            pp_used = int(m.group(2))
            dp_used = int(m.group(3))
            return tp_used, pp_used, dp_used
    return None

def parse_parallel_plan(log_path: str, wait_sec: int = 3, retries: int = 8):
    for _ in range(retries):
        try:
            with open(log_path, "r", errors="ignore") as f:
                text = f.read()
            plan = parse_parallel_plan_from_text(text)
            if plan:
                return plan
        except FileNotFoundError:
            pass
        time.sleep(wait_sec)
    return None


# ----------------- Data classes for grouping -----------------

@dataclass(frozen=True)
class ServerCfg:
    model: str
    host: str
    port: int
    tp: int
    gpu_mem: float
    mns: int
    mnbt: int
    bs: int
    mode: str
    disable_custom_all_reduce: bool
    pp: int          # computed to utilize GPUs (dp stays 1)

@dataclass(frozen=True)
class ClientCfg:
    concurrency: int
    num_requests: int
    max_new_tokens: int
    temperature: float
    prompts_file: Optional[str]
    stream: bool


# ----------------- Main -----------------

def main():
    args = parse_args()
    ensure_dir(args.logdir)
    out_path = Path(args.out)

    # Write header if file doesn't exist
    if not out_path.exists():
        with open(out_path, "w") as f:
            f.write(",".join([
                "timestamp","model","host","port","tp","pp","gpu_mem",
                "max_num_seqs","max_num_batched_tokens","block_size",
                "mode_chunked_prefill","mode_prefix_caching","disable_custom_all_reduce",
                "concurrency","num_requests","max_new_tokens","temperature",
                "success_count","error_count","wall_time_s",
                "total_prompt_tokens","total_completion_tokens","total_tokens",
                "throughput_tokens_per_s","goodput_tokens_per_s","reqs_per_s","notes",
                "ttft_ms_p50","ttft_ms_p95","ttft_ms_p99",
                "tbt_ms_p50","tbt_ms_p95","tbt_ms_p99",
                "log_path",
                "intended_tp","intended_pp","intended_dp",
                "tp_used","pp_used","dp_used"
            ]) + "\n")

    # Build all server-side combos and client-side combos
    ngpus_visible = count_visible_gpus()
    print(f"[info] Visible GPUs = {ngpus_visible} (from CUDA_VISIBLE_DEVICES)")

    server_groups: Dict[ServerCfg, List[ClientCfg]] = defaultdict(list)

    for tp in args.tensor_parallel:
        pp, warn = compute_pp_for_all_gpus(tp, ngpus_visible)
        for gpu_mem in args.gpu_mem:
            for mns in args.max_num_seqs:
                for mnbt in args.max_num_batched_tokens:
                    if mnbt < mns:  # hard constraint
                        continue
                    for bs in args.block_size:
                        for mode in args.modes:
                            mode_chunked = (mode == "chunked")
                            mode_prefix = (mode == "prefix")
                            # Server config / grouping key
                            s_cfg = ServerCfg(
                                model=args.model,
                                host=args.host,
                                port=args.port,
                                tp=tp,
                                gpu_mem=gpu_mem,
                                mns=mns,
                                mnbt=mnbt,
                                bs=bs,
                                mode=mode,
                                disable_custom_all_reduce=args.disable_custom_all_reduce,
                                pp=pp
                            )
                            # Expand client-side sweep (can be multiple; warm start keeps server)
                            for conc in args.concurrency:
                                for nreq in args.num_requests:
                                    for mnt in args.max_new_tokens:
                                        for temp in args.temperature:
                                            for pf in args.prompts_file:
                                                c_cfg = ClientCfg(
                                                    concurrency=conc,
                                                    num_requests=nreq,
                                                    max_new_tokens=mnt,
                                                    temperature=temp,
                                                    prompts_file=pf,
                                                    stream=args.client_stream
                                                )
                                                server_groups[s_cfg].append(c_cfg)

    # Process groups: one server per unique ServerCfg; possibly many client runs each
    total_groups = len(server_groups)
    for gi, (s_cfg, client_list) in enumerate(server_groups.items(), start=1):
        tag = (f"tp{s_cfg.tp}_pp{s_cfg.pp}_mem{s_cfg.gpu_mem}"
               f"_mns{s_cfg.mns}_mnbt{s_cfg.mnbt}_bs{s_cfg.bs}_{s_cfg.mode}")
        log_path = os.path.join(args.logdir, f"serve_{tag}.log")

        # Build server command
        cmd = [
            "vllm", "serve", s_cfg.model,
            "--tensor-parallel-size", str(s_cfg.tp),
            "--pipeline-parallel-size", str(s_cfg.pp),     # PP to utilize GPUs; DP stays 1 (default)
            "--gpu-memory-utilization", str(s_cfg.gpu_mem),
            "--max-num-seqs", str(s_cfg.mns),
            "--max-num-batched-tokens", str(s_cfg.mnbt),
            "--block-size", str(s_cfg.bs),
            "--host", s_cfg.host,
            "--port", str(s_cfg.port),
        ]
        if s_cfg.disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")
        if s_cfg.mode == "chunked":
            cmd.append("--enable-chunked-prefill")
        if s_cfg.mode == "prefix":
            cmd.append("--enable-prefix-caching")

        env = os.environ.copy()

        # ---- Start server (WARM START group) ----
        combo_start_ts = now_epoch()
        proc, lf = start_server(cmd, log_path, env=env)
        lf.write(f"=== group {gi}/{total_groups} :: {tag} ===\n")
        lf.write(f"[intended_plan] TP={s_cfg.tp} PP={s_cfg.pp} DP=1\n")
        lf.flush()
        log_ts(lf, "combo_start", combo_start_ts)
        log_ts(lf, "server_launch_start", combo_start_ts)

        # Wait for readiness once per group
        ready = poll_ready(f"http://{s_cfg.host}:{s_cfg.port}/v1/models", timeout_s=args.timeout_ready)
        ready_ts = now_epoch()
        log_ts(lf, "server_ready", ready_ts, extra=f"elapsed_s={ready_ts - combo_start_ts:.6f}")

        # Parse plan once
        tp_used = pp_used = dp_used = 0
        plan = parse_parallel_plan(log_path)
        if plan:
            tp_used, pp_used, dp_used = plan
            lf.write(f"[plan] Parallel plan detected: TP={tp_used} PP={pp_used} DP={dp_used}\n")
            lf.flush()
        else:
            lf.write("[plan] Parallel plan not detected in log yet.\n")
            lf.flush()

        # If server didn't come up, record failures for all client runs in this group
        if not ready:
            print(f"[error] Server not ready for group {tag}. Skipping its {len(client_list)} client run(s). See {log_path}", file=sys.stderr)
            stop_start = now_epoch()
            log_ts(lf, "server_stop_start", stop_start)
            stop_server(proc, lf)
            with open(log_path, "a") as lf2:
                log_ts(lf2, "server_stop_done", now_epoch(),
                       extra=f"elapsed_s={now_epoch() - stop_start:.6f}")
                log_ts(lf2, "combo_end", now_epoch(), extra="status=not_ready")
            # Write one CSV row per intended client run to indicate failure
            for c_cfg in client_list:
                append_csv_row(
                    out_path, s_cfg, c_cfg, log_path,
                    success_count=0, error_count=c_cfg.num_requests, wall_time_s=0.0,
                    total_prompt_tokens=0, total_completion_tokens=0, total_tokens=0,
                    throughput=0.0, goodput=0.0, rps=0.0,
                    notes="not_ready",
                    ttfts=(0.0, 0.0, 0.0), tbts=(0.0, 0.0, 0.0),
                    intended=(s_cfg.tp, s_cfg.pp, 1),
                    detected=(tp_used, pp_used, dp_used),
                )
            continue

        # ---- Run all client runs under this warm server ----
        for ci, c_cfg in enumerate(client_list, start=1):
            # Client command
            client_cmd = [
                args.python, "llm_load_client.py",
                "--server-url", f"http://{s_cfg.host}:{s_cfg.port}/v1",
                "--model", s_cfg.model,
                "--concurrency", str(c_cfg.concurrency),
                "--num-requests", str(c_cfg.num_requests),
                "--max-new-tokens", str(c_cfg.max_new_tokens),
                "--temperature", str(c_cfg.temperature),
            ]
            if c_cfg.prompts_file:
                client_cmd += ["--prompts-file", c_cfg.prompts_file]
            if c_cfg.stream:
                client_cmd += ["--stream"]

            notes_parts = []
            # If PP had to fall back to 1 due to indivisible, note it
            if s_cfg.pp == 1 and (count_visible_gpus() % s_cfg.tp != 0):
                notes_parts.append(f"pp_warn=not_divisible(visible={count_visible_gpus()},tp={s_cfg.tp})")

            print(f"[info] Group {gi}/{total_groups} :: client {ci}/{len(client_list)} -> {tag} "
                  f"(conc={c_cfg.concurrency}, nreq={c_cfg.num_requests}, mnt={c_cfg.max_new_tokens}, T={c_cfg.temperature})")

            # Log per-client run start
            client_start_ts = now_epoch()
            log_ts(lf, "client_run_start", client_start_ts,
                   extra=f"conc={c_cfg.concurrency} num_req={c_cfg.num_requests} max_new_tokens={c_cfg.max_new_tokens} T={c_cfg.temperature}")

            # Run client
            metrics = {}
            try:
                client_out = subprocess.check_output(client_cmd, stderr=subprocess.STDOUT, timeout=3600)
                client_end_ts = now_epoch()
                log_ts(lf, "client_run_end", client_end_ts,
                       extra=f"elapsed_s={client_end_ts - client_start_ts:.6f}")
                try:
                    metrics = json.loads(client_out.decode("utf-8"))
                except json.JSONDecodeError:
                    print("[error] Failed to parse client JSON. See stdout below:", file=sys.stderr)
                    print(client_out.decode("utf-8"), file=sys.stderr)
                    metrics = default_metrics("client_json_parse_error", c_cfg.num_requests)
            except subprocess.TimeoutExpired:
                client_end_ts = now_epoch()
                log_ts(lf, "client_run_end", client_end_ts,
                       extra=f"elapsed_s={client_end_ts - client_start_ts:.6f} timeout=1")
                metrics = default_metrics("client_timeout", c_cfg.num_requests)
            finally:
                # Optional per-request timestamps
                try:
                    if isinstance(metrics, dict):
                        log_optional_client_marks(lf, metrics)
                except Exception:
                    pass

            # Combine client note with any PP warning
            combined_notes = []
            if metrics.get("notes"):
                combined_notes.append(str(metrics.get("notes")))
            if notes_parts:
                combined_notes.append(" ".join(notes_parts))
            notes_str = " ".join(combined_notes)

            # Append one CSV row for this client run
            append_csv_row(
                out_path, s_cfg, c_cfg, log_path,
                success_count=metrics.get("success_count", 0),
                error_count=metrics.get("error_count", 0),
                wall_time_s=metrics.get("wall_time_s", 0.0),
                total_prompt_tokens=metrics.get("total_prompt_tokens", 0),
                total_completion_tokens=metrics.get("total_completion_tokens", 0),
                total_tokens=metrics.get("total_tokens", 0),
                throughput=metrics.get("throughput_tokens_per_s", 0.0),
                goodput=metrics.get("goodput_tokens_per_s", 0.0),
                rps=metrics.get("reqs_per_s", 0.0),
                notes=notes_str,
                ttfts=(metrics.get("ttft_ms_p50", 0.0), metrics.get("ttft_ms_p95", 0.0), metrics.get("ttft_ms_p99", 0.0)),
                tbts=(metrics.get("tbt_ms_p50", 0.0), metrics.get("tbt_ms_p95", 0.0), metrics.get("tbt_ms_p99", 0.0)),
                intended=(s_cfg.tp, s_cfg.pp, 1),
                detected=(tp_used, pp_used, dp_used),
            )

        # ---- Stop server after all client runs in this group ----
        stop_start = now_epoch()
        log_ts(lf, "server_stop_start", stop_start)
        stop_server(proc, lf)
        with open(log_path, "a") as lf2:
            log_ts(lf2, "server_stop_done", now_epoch(),
                   extra=f"elapsed_s={now_epoch() - stop_start:.6f}")
            log_ts(lf2, "combo_end", now_epoch(), extra="status=ok")

    print(f"[info] Sweep complete. Results in {out_path.resolve()}")


# ----------------- Utilities for rows/metrics -----------------

def default_metrics(reason: str, nreq: int) -> dict:
    return {
        "success_count": 0, "error_count": nreq,
        "wall_time_s": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "throughput_tokens_per_s": 0.0,
        "goodput_tokens_per_s": 0.0,
        "reqs_per_s": 0.0,
        "notes": reason,
        "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0, "ttft_ms_p99": 0.0,
        "tbt_ms_p50": 0.0, "tbt_ms_p95": 0.0, "tbt_ms_p99": 0.0
    }


def append_csv_row(
    out_path: Path,
    s_cfg: ServerCfg,
    c_cfg: ClientCfg,
    log_path: str,
    *,
    success_count: int,
    error_count: int,
    wall_time_s: float,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_tokens: int,
    throughput: float,
    goodput: float,
    rps: float,
    notes: str,
    ttfts: Tuple[float, float, float],
    tbts: Tuple[float, float, float],
    intended: Tuple[int, int, int],
    detected: Tuple[int, int, int],
):
    ts = int(time.time())
    mode_chunked = int(s_cfg.mode == "chunked")
    mode_prefix = int(s_cfg.mode == "prefix")

    row = [
        ts, s_cfg.model, s_cfg.host, s_cfg.port, s_cfg.tp, s_cfg.pp, s_cfg.gpu_mem,
        s_cfg.mns, s_cfg.mnbt, s_cfg.bs,
        mode_chunked, mode_prefix, int(s_cfg.disable_custom_all_reduce),
        c_cfg.concurrency, c_cfg.num_requests, c_cfg.max_new_tokens, c_cfg.temperature,
        success_count, error_count, wall_time_s,
        total_prompt_tokens, total_completion_tokens, total_tokens,
        throughput, goodput, rps, notes,
        ttfts[0], ttfts[1], ttfts[2],
        tbts[0], tbts[1], tbts[2],
        log_path,
        intended[0], intended[1], intended[2],
        detected[0], detected[1], detected[2],
    ]
    with open(out_path, "a") as f:
        f.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
