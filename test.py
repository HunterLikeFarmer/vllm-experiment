# test.py  (Option 2: TP=2 allowed, inject safe NCCL env for TP>1)
import subprocess
import time
import pandas as pd
import asyncio
import sys
import argparse
import itertools
import os
from datetime import datetime

# Import the benchmark function
sys.path.insert(0, '.')
from my_benchmark import benchmark


def gpu_count():
    """Return the number of visible CUDA GPUs (best-effort)."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        # Fallback: try nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL
            ).decode().strip().splitlines()
            return len(out)
        except Exception:
            return 0


def generate_configurations(include_quantized=False):
    """Generate test configurations for vLLM 0.11.0 (Option 2)."""

    # vLLM 0.11.0 supported parameters
    base_params = {
        'tensor_parallel': [1, 2],  # will be filtered by actual GPU count
        'max_num_seqs': [64, 256, 1024, 4096],
        'max_num_batched_tokens': [64, 512, 2048, 8192],
        'block_size': [8, 16, 32],
        'enable_chunked_prefill': [True, False],
        'enable_prefix_caching': [True, False],
        'disable_custom_all_reduce': [True, False],
        'gpu_memory_utilization': [0.80, 0.90],
    }

    base_configs = []
    keys = list(base_params.keys())

    total_combinations = 1
    for k in keys:
        total_combinations *= len(base_params[k])
    print(f"Total possible combinations: {total_combinations}")

    visible_gpus = max(1, gpu_count())
    print(f"Visible GPUs detected: {visible_gpus}")

    for values in itertools.product(*[base_params[k] for k in keys]):
        config = dict(zip(keys, values))

        # Constraints
        if config['max_num_batched_tokens'] < config['max_num_seqs']:
            continue
        if config['enable_chunked_prefill'] and config['enable_prefix_caching']:
            continue
        if config['tensor_parallel'] > visible_gpus:
            continue

        config['model'] = 'Qwen/Qwen3-8B'
        # Ensure the served model name equals what the client sends
        config['served_name'] = config['model']
        config['quantization'] = None

        base_configs.append(config)

    print(f"Generated {len(base_configs)} valid base configurations after constraints")

    if include_quantized:
        quant_params = {
            'tensor_parallel': [1, 2],
            'max_num_seqs': [64, 256, 1024],
            'max_num_batched_tokens': [512, 2048],
            'block_size': [16],  # fixed for quantized run
            'enable_chunked_prefill': [False],
            'enable_prefix_caching': [True, False],
            'disable_custom_all_reduce': [True, False],
            'gpu_memory_utilization': [0.80, 0.90],
        }
        quant_configs = []
        keys = list(quant_params.keys())
        for values in itertools.product(*[quant_params[k] for k in keys]):
            config = dict(zip(keys, values))
            if config['max_num_batched_tokens'] < config['max_num_seqs']:
                continue
            if config['tensor_parallel'] > visible_gpus:
                continue
            # Choose ONE approach for quantization:
            # Either a base model + --quantization, or a pre-quantized repo (without the flag).
            config['model'] = 'Qwen/Qwen3-8B'
            config['served_name'] = config['model']
            config['quantization'] = 'awq'
            quant_configs.append(config)
        print(f"Generated {len(quant_configs)} quantized configurations")
        return base_configs + quant_configs

    return base_configs


def start_vllm_server(config, port):
    """Start vLLM server with given configuration (inject safe NCCL env for TP>1)."""

    cmd = [
        'vllm', 'serve', config['model'],
        '--served-model-name', config.get('served_name', config['model']),
        '--tensor-parallel-size', str(config['tensor_parallel']),
        '--gpu-memory-utilization', str(config['gpu_memory_utilization']),
        '--max-num-seqs', str(config['max_num_seqs']),
        '--max-num-batched-tokens', str(config['max_num_batched_tokens']),
        '--block-size', str(config['block_size']),
        '--host', '0.0.0.0', '--port', str(port),
        '--disable-log-requests',  # less noisy server logs
    ]

    # Mutually exclusive flags (only add if True)
    if config['enable_chunked_prefill']:
        cmd.append('--enable-chunked-prefill')
    if config['enable_prefix_caching']:
        cmd.append('--enable-prefix-caching')
    if config['disable_custom_all_reduce']:
        cmd.append('--disable-custom-all-reduce')

    # Quantization (only if specified)
    if config.get('quantization'):
        cmd.extend(['--quantization', config['quantization']])

    # Build a safe env for NCCL when TP>1
    env = os.environ.copy()
    if config.get('tensor_parallel', 1) > 1:
        env.setdefault('NCCL_DEBUG', 'INFO')
        env.setdefault('NCCL_IB_DISABLE', '1')   # avoid RDMA path
        env.setdefault('NCCL_P2P_DISABLE', '1')  # avoid GPU P2P requirement
        # Optional if your host has many virtual interfaces:
        # env.setdefault('NCCL_SOCKET_IFNAME', '^lo,docker,veth')

    print(f"\nStarting server with command:\n  {' '.join(cmd)}\n")
    # Avoid PIPE deadlocks: inherit parent stdout/stderr
    proc = subprocess.Popen(cmd, env=env)
    return proc


def wait_for_server(host, port, timeout=180):
    """Wait for server to be ready (/health then /v1/models)."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        for path in ("/health", "/v1/models"):
            try:
                r = requests.get(f"http://{host}:{port}{path}", timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
        time.sleep(2)
    return False


def run_benchmark_for_config(config, port, benchmark_params):
    """Run benchmark for a single configuration."""
    try:
        result = asyncio.run(benchmark(
            host="localhost",
            port=port,
            model=config['served_name'],  # matches --served-model-name
            num_requests=benchmark_params['num_requests'],
            input_len=benchmark_params['input_len'],
            output_len=benchmark_params['output_len'],
            request_rate=benchmark_params['request_rate']
        ))
        return {
            'success': True,
            'throughput': result['throughput'],
            'token_throughput': result['token_throughput'],
            'mean_latency': result['mean_latency'],
            'p50_latency': result['p50_latency'],
            'p90_latency': result['p90_latency'],
            'p99_latency': result['p99_latency'],
            'total_time': result['total_time'],
            'successful_requests': result['successful_requests'],
            'failed_requests': result['failed_requests'],
        }
    except Exception as e:
        print(f"Benchmark error: {e}")
        return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='vLLM 0.11.0 configuration testing (Option 2: TP=2 with safe NCCL env)'
    )
    parser.add_argument('--include-quantized', action='store_true',
                        help='Include quantized model tests (AWQ)')
    parser.add_argument('--num-requests', type=int, default=50,
                        help='Number of requests per benchmark')
    parser.add_argument('--input-len', type=int, default=512,
                        help='Input token length')
    parser.add_argument('--output-len', type=int, default=128,
                        help='Output token length')
    parser.add_argument('--request-rate', type=float, default=None,
                        help='Request rate (None for unlimited)')
    parser.add_argument('--start-port', type=int, default=8000,
                        help='Starting port number')
    parser.add_argument('--output-file', type=str, default='vllm_results.csv',
                        help='Output CSV file')
    parser.add_argument('--skip-warmup', action='store_true',
                        help='Skip server warmup wait')
    parser.add_argument('--sample-configs', type=int, default=None,
                        help='Randomly sample N configs to test')
    parser.add_argument('--yes', action='store_true',
                        help='Proceed without confirmation prompt')
    args = parser.parse_args()

    # Generate configurations
    print("=" * 80)
    print("GENERATING TEST CONFIGURATIONS (vLLM 0.11.0 / TP safe NCCL env when TP>1)")
    print("=" * 80)
    configs = generate_configurations(include_quantized=args.include_quantized)

    # Sample configs if requested
    if args.sample_configs and args.sample_configs < len(configs):
        import random
        random.seed(42)
        configs = random.sample(configs, args.sample_configs)
        print(f"\nRandomly sampled {len(configs)} configurations for testing")

    print(f"\nTotal configurations to test: {len(configs)}")
    print(f"Estimated time: ~{len(configs) * 3} minutes (assuming 3 min per config)")

    # Confirmation (robust for non-interactive shells)
    if not args.yes:
        proceed = True if not sys.stdin.isatty() else None
        if proceed is None:
            try:
                response = input("\nProceed with testing? (y/n): ").strip().lower()
                proceed = (response in ('y', 'yes'))
            except EOFError:
                proceed = False
        if not proceed:
            print("Testing cancelled.")
            return

    benchmark_params = {
        'num_requests': args.num_requests,
        'input_len': args.input_len,
        'output_len': args.output_len,
        'request_rate': args.request_rate,
    }

    results = []
    port = args.start_port
    start_time = datetime.now()

    for idx, config in enumerate(configs, 1):
        print("\n" + "=" * 80)
        print(f"TEST {idx}/{len(configs)}")
        print("=" * 80)
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"Port: {port}")

        # Start server
        print("\nStarting vLLM server...")
        server_proc = start_vllm_server(config, port)

        # Warmup
        warmup_time = 30 if args.skip_warmup else 90
        print(f"Waiting {warmup_time}s for server to be ready...")
        time.sleep(warmup_time)

        # Readiness check (best effort)
        if not wait_for_server("localhost", port, timeout=120):
            print("WARNING: Server may not be ready, proceeding with benchmark...")

        # Run benchmark
        print("\nRunning benchmark...")
        bench_result = run_benchmark_for_config(config, port, benchmark_params)

        # Combine config and results
        result_row = {**config}
        if bench_result.get('success'):
            result_row.update({
                'throughput_req_per_s': bench_result['throughput'],
                'token_throughput': bench_result['token_throughput'],
                'mean_latency_s': bench_result['mean_latency'],
                'p50_latency_s': bench_result['p50_latency'],
                'p90_latency_s': bench_result['p90_latency'],
                'p99_latency_s': bench_result['p99_latency'],
                'total_time_s': bench_result['total_time'],
                'successful_requests': bench_result['successful_requests'],
                'failed_requests': bench_result['failed_requests'],
                'status': 'SUCCESS',
            })
        else:
            result_row.update({
                'status': 'FAILED',
                'error': bench_result.get('error', 'Unknown error'),
            })

        results.append(result_row)

        # Stop server
        print("\nStopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()

        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        print(f"\nIntermediate results saved to {args.output_file}")

        # Wait between tests
        print("\nWaiting 10s before next test...")
        time.sleep(10)

        port += 1

    # Final results
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print(f"Total time: {duration}")
    print(f"Results saved to: {args.output_file}")

    # Summary
    df = pd.DataFrame(results)
    successful = df[df['status'] == 'SUCCESS']

    print(f"\nSummary:")
    print(f"  Total configurations: {len(configs)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(df) - len(successful)}")

    if len(successful) > 0:
        print("\nTop 5 configurations by throughput:")
        top5 = successful.nlargest(5, 'throughput_req_per_s')[
            ['tensor_parallel', 'max_num_seqs', 'max_num_batched_tokens',
             'gpu_memory_utilization', 'enable_chunked_prefill', 'enable_prefix_caching',
             'throughput_req_per_s', 'token_throughput', 'p99_latency_s']
        ]
        print(top5.to_string(index=False))

        print("\nTop 5 configurations by token throughput:")
        top5_tokens = successful.nlargest(5, 'token_throughput')[
            ['tensor_parallel', 'max_num_seqs', 'max_num_batched_tokens',
             'gpu_memory_utilization', 'enable_chunked_prefill', 'enable_prefix_caching',
             'throughput_req_per_s', 'token_throughput', 'p99_latency_s']
        ]
        print(top5_tokens.to_string(index=False))

    # List all configurations tested
    print("\n" + "=" * 80)
    print("ALL CONFIGURATIONS TESTED")
    print("=" * 80)
    for i, config in enumerate(configs, 1):
        status = results[i - 1].get('status', 'UNKNOWN')
        throughput = results[i - 1].get('throughput_req_per_s', 'N/A')
        print(f"\n{i}. TP={config['tensor_parallel']}, "
              f"GPU_MEM={config['gpu_memory_utilization']}, "
              f"MAX_SEQS={config['max_num_seqs']}, "
              f"MAX_TOKENS={config['max_num_batched_tokens']}")
        print(f"   block_size={config['block_size']}")
        print(f"   chunked_prefill={config['enable_chunked_prefill']}, "
              f"prefix_cache={config['enable_prefix_caching']}")
        print(f"   disable_all_reduce={config['disable_custom_all_reduce']}")
        print(f"   model={config['model']}, quant={config['quantization']}")
        print(f"   Status: {status}, Throughput: {throughput}")


if __name__ == "__main__":
    main()

# python test.py --sample-configs 10 --num-requests 20 --input-len 512 --output-len 128 --yes
