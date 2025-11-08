"""
run_vllm_sweep.py
Orchestrates a sweep of vLLM server parameters and records throughput/goodput (+ TTFT/TBT if client streams).

Adds detailed timestamp logging to each run's log file, including:
- combo start/end
- server launch start
- server ready (elapsed)
- client run start/end (elapsed)
- server stop start/done (elapsed)
- optional first/last request timestamps if provided by llm_load_client.py JSON

This version:
- Always enforces Data Parallelism (DP) = 1 (single server process).
- Computes Pipeline Parallelism (PP) so that TP * PP == #visible GPUs (from CUDA_VISIBLE_DEVICES).
- Passes --pipeline-parallel-size <PP> to vLLM.
- Logs the intended (TP,PP,DP=1) and the detected actual plan parsed from vLLM logs.
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
from datetime import datetime, timezone
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


def count_visible_gpus() -> int:
    """
    Count GPUs from CUDA_VISIBLE_DEVICES if present; otherwise fall back to 1.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip() != ""])
    return 1


def choose_pp_for_all_gpus(tp: int, ngpus: int):
    """
    Compute PP so that tp * pp == ngpus with DP fixed at 1.
    Returns (pp, warn_msg). If ngpus % tp != 0, use pp = ngpus // tp and warn.
    """
    if tp <= 0 or ngpus <= 0:
        return 1, f"invalid(tp={tp},ngpus={ngpus})"
    if ngpus % tp == 0:
        return ngpus // tp, ""
    pp = max(1, ngpus // tp)
    unused = ngpus - tp * pp
    return pp, f"not_divisible(ngpus={ngpus},tp={tp})->pp={pp},unused_gpus={unused}"


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
        # Log key env for traceability
        for k in ["CUDA_VISIBLE_DEVICES"]:
            if k in env:
                lf.write(f"[env] {k}={env[k]}\n")
    # POSIX process group so we can terminate child tree
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


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# -------- Timestamp helpers --------

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


def main():
    args = parse_args()
    ensure_dir(args.logdir)
    out_path = Path(args.out)

    # CSV header
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
                "log_path",
                "intended_tp","intended_pp","intended_dp",  # what we asked for
                "tp_used","pp_used","dp_used"                # what vLLM actually used (parsed)
            ]) + "\n")

    # Build combos
    combos = []
    for tp in args.tensor_parallel:
        for gpu_mem in args.gpu_mem:
            for mns in args.max_num_seqs:
                for mnbt in args.max_num_batched_tokens:
                    if mnbt < mns:
                        continue
                    for bs in args.block_size:
                        for mode in args.modes:
                            combos.append((tp, gpu_mem, mns, mnbt, bs, mode))

    if not combos:
        print("[error] No valid parameter combinations after constraints.", file=sys.stderr)
        sys.exit(2)

    ngpus_visible = count_visible_gpus()
    print(f"[info] Visible GPUs = {ngpus_visible} (from CUDA_VISIBLE_DEVICES)")
    if ngpus_visible <= 0:
        print("[error] No visible GPUs; check CUDA_VISIBLE_DEVICES.", file=sys.stderr)
        sys.exit(2)

    for idx, (tp, gpu_mem, mns, mnbt, bs, mode) in enumerate(combos, start=1):
        ts = int(time.time())
        tag = f"tp{tp}_mem{gpu_mem}_mns{mns}_mnbt{mnbt}_bs{bs}_{mode}"
        log_path = os.path.join(args.logdir, f"serve_{tag}.log")

        # Decide PP with DP fixed at 1
        intended_dp = 1
        intended_pp, pp_warn = choose_pp_for_all_gpus(tp, ngpus_visible)
        notes_parts = []
        if pp_warn:
            notes_parts.append(f"pp_warn={pp_warn}")
        if tp * intended_pp != ngpus_visible:
            notes_parts.append(f"partial_use(tp={tp},pp={intended_pp},visible={ngpus_visible})")

        cmd = [
            "vllm", "serve", args.model,
            "--tensor-parallel-size", str(tp),
            "--pipeline-parallel-size", str(intended_pp),  # PP computed here
            "--gpu-memory-utilization", str(gpu_mem),
            "--max-num-seqs", str(mns),
            "--max-num-batched-tokens", str(mnbt),
            "--block-size", str(bs),
            "--host", args.host,
            "--port", str(args.port),
        ]
        if args.disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")

        mode_chunked = (mode == "chunked")
        mode_prefix = (mode == "prefix")
        if mode_chunked:
            cmd.append("--enable-chunked-prefill")
        if mode_prefix:
            cmd.append("--enable-prefix-caching")

        env = os.environ.copy()  # DP is implicitly 1 (single process)

        # Start server
        combo_start_ts = now_epoch()
        proc, lf = start_server(cmd, log_path, env=env)
        lf.write(f"=== combo {idx}/{len(combos)} :: {tag} ===\n")
        lf.write(f"[intended_plan] TP={tp} PP={intended_pp} DP=1 (single process)\n")
        lf.flush()
        log_ts(lf, "combo_start", combo_start_ts)
        log_ts(lf, "server_launch_start", combo_start_ts)

        # Wait for readiness
        ready = poll_ready(f"http://{args.host}:{args.port}/v1/models", timeout_s=args.timeout_ready)
        ready_ts = now_epoch()
        log_ts(lf, "server_ready", ready_ts, extra=f"elapsed_s={ready_ts - combo_start_ts:.6f}")

        # Parse actual plan
        tp_used = pp_used = dp_used = 0
        plan = parse_parallel_plan(log_path)
        if plan:
            tp_used, pp_used, dp_used = plan
            lf.write(f"[plan] Parallel plan detected: TP={tp_used} PP={pp_used} DP={dp_used}\n")
            lf.flush()
        else:
            lf.write("[plan] Parallel plan not detected in log yet.\n")
            lf.flush()

        if not ready:
            print(f"[error] Server not ready for combo {tag}. Skipping. See log: {log_path}", file=sys.stderr)
            stop_start = now_epoch()
            log_ts(lf, "server_stop_start", stop_start)
            stop_server(proc, lf)

            with open(out_path, "a") as f:
                f.write(",".join(map(str, [
                    ts,args.model,args.host,args.port,tp,gpu_mem,
                    mns,mnbt,bs,
                    int(mode_chunked),int(mode_prefix),int(args.disable_custom_all_reduce),
                    args.concurrency,args.num_requests,args.max_new_tokens,args.temperature,
                    0,1,0.0,0,0,0,0.0,0.0,0.0,"not_ready" + ((" " + " ".join(notes_parts)) if notes_parts else ""),
                    0.0,0.0,0.0, 0.0,0.0,0.0,
                    log_path,
                    tp, intended_pp, 1,         # intended
                    tp_used, pp_used, dp_used   # actual
                ])) + "\n")

            with open(log_path, "a") as lf2:
                log_ts(lf2, "server_stop_done", now_epoch(), extra=f"elapsed_s={now_epoch() - stop_start:.6f}")
                log_ts(lf2, "combo_end", now_epoch(), extra="status=not_ready")
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
        client_start_ts = now_epoch()
        log_ts(lf, "client_run_start", client_start_ts)

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
            client_end_ts = now_epoch()
            log_ts(lf, "client_run_end", client_end_ts,
                   extra=f"elapsed_s={client_end_ts - client_start_ts:.6f} timeout=1")
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
            try:
                if isinstance(metrics, dict):
                    log_optional_client_marks(lf, metrics)
            except Exception:
                pass
            stop_start = now_epoch()
            log_ts(lf, "server_stop_start", stop_start)
            stop_server(proc, lf)
            with open(log_path, "a") as lf2:
                log_ts(lf2, "server_stop_done", now_epoch(),
                       extra=f"elapsed_s={now_epoch() - stop_start:.6f}")
                log_ts(lf2, "combo_end", now_epoch(), extra="status=ok")

        # Notes
        combined_notes = []
        if metrics.get("notes"):
            combined_notes.append(str(metrics.get("notes")))
        if notes_parts:
            combined_notes.append(" ".join(notes_parts))
        notes_str = " ".join(combined_notes)

        # Output row
        with open(out_path, "a") as f:
            f.write(",".join(map(str, [
                ts,args.model,args.host,args.port,tp,gpu_mem,
                mns,mnbt,bs,
                int(mode_chunked),int(mode_prefix),int(args.disable_custom_all_reduce),
                args.concurrency,args.num_requests,args.max_new_tokens,args.temperature,
                metrics.get("success_count",0),metrics.get("error_count",0),metrics.get("wall_time_s",0.0),
                metrics.get("total_prompt_tokens",0),metrics.get("total_completion_tokens",0),metrics.get("total_tokens",0),
                metrics.get("throughput_tokens_per_s",0.0),metrics.get("goodput_tokens_per_s",0.0),metrics.get("reqs_per_s",0.0),
                notes_str,
                metrics.get("ttft_ms_p50",0.0),metrics.get("ttft_ms_p95",0.0),metrics.get("ttft_ms_p99",0.0),
                metrics.get("tbt_ms_p50",0.0),metrics.get("tbt_ms_p95",0.0),metrics.get("tbt_ms_p99",0.0),
                log_path,
                tp, intended_pp, 1,          # intended (DP=1)
                tp_used, pp_used, dp_used    # detected actual
            ])) + "\n")

        print(f"[info] Done combo {idx}/{len(combos)} -> {tag}")

    print(f"[info] Sweep complete. Results in {out_path.resolve()}")


if __name__ == "__main__":
    main()
