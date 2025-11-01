#!/usr/bin/env python3
"""
llm_load_client.py
Simple concurrent client for an OpenAI-compatible server (e.g., vLLM `vllm serve`).

Outputs a single JSON object to stdout with aggregate metrics:
{
  "success_count": int,
  "error_count": int,
  "wall_time_s": float,
  "total_prompt_tokens": int,
  "total_completion_tokens": int,
  "total_tokens": int,
  "throughput_tokens_per_s": float,
  "goodput_tokens_per_s": float,
  "reqs_per_s": float,
  "notes": str,
  "latency_ms_p50": float,
  "latency_ms_p95": float,
  "latency_ms_p99": float,
  "ttft_ms_p50": float,
  "ttft_ms_p95": float,
  "ttft_ms_p99": float,
  "tbt_ms_p50": float,
  "tbt_ms_p95": float,
  "tbt_ms_p99": float,
  "status_counts": {"200": int, "429": int, ...},
  "mode": "closed",
  "streaming": bool
}

Notes:
- Use --stream to measure true TTFT. In streaming mode we set
  stream_options.include_usage=true and parse usage from SSE.
- TBT (time-between-tokens) ≈ (latency - TTFT) / completion_tokens
  for each successful streaming request.
"""

import argparse
import asyncio
import json
import time
from typing import List, Optional

import aiohttp

DEFAULT_PROMPTS = [
    "Write a 3-sentence summary about the benefits and risks of large language models in customer support.",
    "Give 5 bullet points on how to optimize Python code for speed without changing algorithms.",
    "Explain in simple terms how attention works in transformers.",
    "Summarize the current best practices for prompt engineering to reduce hallucinations.",
    "Provide a concise overview of the Qwen 3 family and common use cases.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://127.0.0.1:8000/v1",
                   help="Base URL to the OpenAI-compatible API (no trailing slash beyond /v1)")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompts-file", type=str, default=None)

    # Streaming & extras
    p.add_argument("--stream", action="store_true",
                   help="Use streaming; enables true TTFT measurement.")
    p.add_argument("--seed", type=int, default=42,
                   help="Forwarded to server if it's supported.")
    p.add_argument("--warmup", type=int, default=0,
                   help="Warmup requests (not counted in metrics).")
    p.add_argument("--system-file", type=str, default=None,
                   help="Optional system/prefix message (string content).")
    return p.parse_args()


def load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS
    prompts = []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if s:
                prompts.append(s)
    return prompts or DEFAULT_PROMPTS


async def one_request(session: aiohttp.ClientSession, url: str, model: str, prompt: str,
                      max_new_tokens: int, temperature: float, stream: bool,
                      seed: int, system_msg: Optional[str]):
    payload = {
        "model": model,
        "messages": ([{"role": "system", "content": system_msg}] if system_msg else []) +
                    [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": stream,
        "seed": seed
    }
    # Ask the server to include token usage in the final SSE chunk (OpenAI-compatible option).
    if stream:
        payload["stream_options"] = {"include_usage": True}

    try:
        t0 = time.perf_counter()
        async with session.post(f"{url}/chat/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=300)) as resp:
            # Non-streaming path
            if not stream:
                if resp.status != 200:
                    txt = await resp.text()
                    t1 = time.perf_counter()
                    return {"ok": False, "status": resp.status, "error": f"HTTP {resp.status}: {txt}",
                            "pt": 0, "ct": 0, "tt": 0,
                            "lat_ms": (t1 - t0) * 1000.0, "ttft_ms": None, "tbt_ms": None}
                data = await resp.json()
                t1 = time.perf_counter()
                usage = data.get("usage", {}) or {}
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                tt = int(usage.get("total_tokens", pt + ct))
                return {"ok": True, "status": resp.status, "pt": pt, "ct": ct, "tt": tt,
                        "lat_ms": (t1 - t0) * 1000.0, "ttft_ms": None, "tbt_ms": None}

            # Streaming path (SSE). Compute TTFT from first token, parse usage from SSE events.
            ttft = None
            last_usage = {}
            # We'll parse 'data: {...}\n' lines from chunks.
            buffer = b""
            async for raw, _ in resp.content.iter_chunks():
                if not raw:
                    continue
                buffer += raw
                # Process complete lines; keep any partial line in buffer.
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = rest
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload_bytes = line[5:].strip()  # after 'data:'
                    if not payload_bytes or payload_bytes == b"[DONE]":
                        continue
                    # First token? (heuristic: first SSE JSON with choices/delta/content)
                    if ttft is None:
                        try:
                            j0 = json.loads(payload_bytes.decode())
                            ch = (j0.get("choices") or [{}])[0]
                            delta = ch.get("delta") or {}
                            if "content" in delta and delta.get("content") is not None:
                                ttft = (time.perf_counter() - t0) * 1000.0
                        except Exception:
                            pass
                    # Capture usage if present (usually only in the final SSE object)
                    try:
                        j = json.loads(payload_bytes.decode())
                        if "usage" in j and j["usage"]:
                            last_usage = j["usage"] or {}
                    except Exception:
                        pass

            t1 = time.perf_counter()
            if resp.status != 200:
                return {"ok": False, "status": resp.status,
                        "error": b"(stream error)".decode(errors="ignore"),
                        "pt": 0, "ct": 0, "tt": 0,
                        "lat_ms": (t1 - t0) * 1000.0, "ttft_ms": None, "tbt_ms": None}

            pt = int(last_usage.get("prompt_tokens", 0))
            ct = int(last_usage.get("completion_tokens", 0))
            tt = int(last_usage.get("total_tokens", pt + ct)) if (pt or ct) else pt + ct
            lat_ms = (t1 - t0) * 1000.0
            tbt_ms = (max(lat_ms - (ttft or 0.0), 0.0) / ct) if (ttft is not None and ct > 0) else None
            return {"ok": True, "status": resp.status, "pt": pt, "ct": ct, "tt": tt,
                    "lat_ms": lat_ms, "ttft_ms": ttft, "tbt_ms": tbt_ms}

    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e), "pt": 0, "ct": 0, "tt": 0,
                "lat_ms": None, "ttft_ms": None, "tbt_ms": None}


async def run_load(server_url: str, model: str, prompts: List[str], concurrency: int, num_requests: int,
                   max_new_tokens: int, temperature: float, stream: bool,
                   seed: int, warmup: int, system_msg: Optional[str]):
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def task_runner(i: int):
            prompt = prompts[i % len(prompts)]
            async with sem:
                return await one_request(session, server_url, model, prompt, max_new_tokens,
                                         temperature, stream, seed, system_msg)

        # Warmup (not counted)
        if warmup > 0:
            await asyncio.gather(*(task_runner(i) for i in range(warmup)))

        t0 = time.time()
        results = await asyncio.gather(*(task_runner(i) for i in range(num_requests)))
        wall = time.time() - t0

    success = sum(1 for r in results if r["ok"])
    errors = num_requests - success
    total_pt = sum(r["pt"] for r in results)
    total_ct = sum(r["ct"] for r in results)
    total_tt = sum(r["tt"] for r in results)

    # Percentiles
    def percentile(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        k = max(0, min(len(vals) - 1, int(q * (len(vals) - 1))))
        return float(vals[k])

    lats = [r["lat_ms"] for r in results if r.get("lat_ms") is not None]
    ttfts = [r["ttft_ms"] for r in results if r["ok"] and r.get("ttft_ms") is not None]
    tbts = [r["tbt_ms"] for r in results if r["ok"] and r.get("tbt_ms") is not None]

    status_counts = {}
    for r in results:
        s = str(r.get("status", 0))
        status_counts[s] = status_counts.get(s, 0) + 1

    throughput = (total_ct / wall) if wall > 0 else 0.0
    goodput = (sum(r["ct"] for r in results if r["ok"]) / wall) if wall > 0 else 0.0
    rps = (num_requests / wall) if wall > 0 else 0.0

    out = {
        "success_count": success,
        "error_count": errors,
        "wall_time_s": wall,
        "total_prompt_tokens": total_pt,
        "total_completion_tokens": total_ct,
        "total_tokens": total_tt,
        "throughput_tokens_per_s": throughput,
        "goodput_tokens_per_s": goodput,
        "reqs_per_s": rps,
        "notes": "",
        "latency_ms_p50": percentile(lats, 0.50),
        "latency_ms_p95": percentile(lats, 0.95),
        "latency_ms_p99": percentile(lats, 0.99),
        "ttft_ms_p50": percentile(ttfts, 0.50),
        "ttft_ms_p95": percentile(ttfts, 0.95),
        "ttft_ms_p99": percentile(ttfts, 0.99),
        "tbt_ms_p50": percentile(tbts, 0.50),
        "tbt_ms_p95": percentile(tbts, 0.95),
        "tbt_ms_p99": percentile(tbts, 0.99),
        "status_counts": status_counts,
        "mode": "closed",
        "streaming": stream,
    }
    print(json.dumps(out))


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    system_msg = None
    if args.system_file:
        try:
            with open(args.system_file, "r") as f:
                system_msg = f.read()
        except Exception:
            system_msg = None

    asyncio.run(run_load(
        server_url=args.server_url,
        model=args.model,
        prompts=prompts,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stream=args.stream,
        seed=args.seed,
        warmup=args.warmup,
        system_msg=system_msg
    ))


if __name__ == "__main__":
    main()
