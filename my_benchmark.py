import asyncio
import aiohttp
import time
import numpy as np
from typing import List

async def send_request(session, url, model, input_len, output_len, request_id):
    """Send a single request to the vLLM server."""
    payload = {
        "model": model,
        "prompt": "test " * input_len,  # Approximate token count
        "max_tokens": output_len,
        "temperature": 0.7,
    }
    
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            await response.json()
            latency = time.time() - start_time
            return {"success": True, "latency": latency, "request_id": request_id}
    except Exception as e:
        return {"success": False, "error": str(e), "request_id": request_id}

async def benchmark(host, port, model, num_requests, input_len, output_len, request_rate=None):
    """Run benchmark with specified parameters."""
    url = f"http://{host}:{port}/v1/completions"
    
    print(f"Starting benchmark:")
    print(f"  Total requests: {num_requests}")
    print(f"  Input length: ~{input_len} tokens")
    print(f"  Output length: {output_len} tokens")
    print(f"  Request rate: {'unlimited' if request_rate is None else f'{request_rate} req/s'}")
    print()
    
    connector = aiohttp.TCPConnector(limit=1000)
    timeout = aiohttp.ClientTimeout(total=600)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        start_time = time.time()
        
        for i in range(num_requests):
            task = send_request(session, url, model, input_len, output_len, i)
            tasks.append(task)
            
            # Rate limiting
            if request_rate is not None and i < num_requests - 1:
                await asyncio.sleep(1.0 / request_rate)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)
        end_time = time.time()
    
    # Calculate metrics
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if not successful_results:
        print("All requests failed!")
        return
    
    latencies = [r["latency"] for r in successful_results]
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful requests: {len(successful_results)}/{num_requests}")
    print(f"Failed requests: {len(failed_results)}")
    print(f"\nThroughput: {len(successful_results) / total_time:.2f} requests/s")
    print(f"Total tokens generated: ~{len(successful_results) * output_len}")
    print(f"Token throughput: ~{len(successful_results) * output_len / total_time:.2f} tokens/s")
    print(f"\nLatency Statistics:")
    print(f"  Mean: {np.mean(latencies):.3f}s")
    print(f"  Median (P50): {np.percentile(latencies, 50):.3f}s")
    print(f"  P90: {np.percentile(latencies, 90):.3f}s")
    print(f"  P95: {np.percentile(latencies, 95):.3f}s")
    print(f"  P99: {np.percentile(latencies, 99):.3f}s")
    print(f"  Min: {np.min(latencies):.3f}s")
    print(f"  Max: {np.max(latencies):.3f}s")
    print("="*60)
    
    return {
        "throughput": len(successful_results) / total_time,
        "token_throughput": len(successful_results) * output_len / total_time,
        "mean_latency": np.mean(latencies),
        "p50_latency": np.percentile(latencies, 50),
        "p90_latency": np.percentile(latencies, 90),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "successful_requests": len(successful_results),
        "failed_requests": len(failed_results)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=None, 
                        help="Request rate in req/s. None for unlimited.")
    
    args = parser.parse_args()
    
    asyncio.run(benchmark(
        args.host,
        args.port,
        args.model,
        args.num_prompts,
        args.input_len,
        args.output_len,
        args.request_rate
    ))