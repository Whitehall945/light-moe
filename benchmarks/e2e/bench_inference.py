"""
End-to-End Inference Benchmark

Measures:
- Throughput (tokens/second)
- Latency (TTFT, TPOT)
- Memory usage
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    
    model_name: str
    batch_size: int
    input_length: int
    output_length: int
    
    # Performance metrics
    total_time_s: float
    throughput_tokens_per_s: float
    time_to_first_token_ms: float
    time_per_output_token_ms: float
    
    # Memory metrics
    peak_memory_gb: float
    model_memory_gb: float
    
    # Configuration
    num_gpus: int
    expert_parallel_size: int
    dtype: str


def measure_memory() -> float:
    """Get current GPU memory usage in GB."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def benchmark_inference(
    model,
    tokenizer,
    prompts: list[str],
    output_length: int = 128,
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> BenchmarkResult:
    """
    Benchmark inference performance.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        prompts: List of prompts
        output_length: Number of tokens to generate
        warmup_iterations: Warmup iterations
        benchmark_iterations: Benchmark iterations
    """
    batch_size = len(prompts)
    
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].cuda()
    input_length = input_ids.size(1)
    
    # Warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=output_length,
                do_sample=False,
            )
    
    # Measure model memory
    model_memory = measure_memory()
    
    # Benchmark with detailed timing
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    ttft_times = []
    tpot_times = []
    total_times = []
    
    for _ in range(benchmark_iterations):
        # Time to first token
        start = time.perf_counter()
        
        with torch.no_grad():
            # Prefill
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
        
        torch.cuda.synchronize()
        ttft = time.perf_counter() - start
        
        # Generate remaining tokens
        current_ids = input_ids
        decode_start = time.perf_counter()
        
        for _ in range(output_length):
            with torch.no_grad():
                outputs = model(current_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                if hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                
                # Sample next token (greedy)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
        
        total_time = ttft + decode_time
        tpot = decode_time / output_length * 1000
        
        ttft_times.append(ttft * 1000)
        tpot_times.append(tpot)
        total_times.append(total_time)
    
    # Calculate statistics
    avg_ttft = sum(ttft_times) / len(ttft_times)
    avg_tpot = sum(tpot_times) / len(tpot_times)
    avg_total = sum(total_times) / len(total_times)
    
    total_tokens = batch_size * output_length
    throughput = total_tokens / avg_total
    
    peak_memory = measure_memory()
    
    return BenchmarkResult(
        model_name=getattr(model, 'name', 'unknown'),
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
        total_time_s=avg_total,
        throughput_tokens_per_s=throughput,
        time_to_first_token_ms=avg_ttft,
        time_per_output_token_ms=avg_tpot,
        peak_memory_gb=peak_memory,
        model_memory_gb=model_memory,
        num_gpus=torch.cuda.device_count(),
        expert_parallel_size=getattr(model, 'expert_parallel_size', 1),
        dtype=str(next(model.parameters()).dtype),
    )


def run_benchmark_suite(
    model_path: str,
    batch_sizes: list[int] = [1, 4, 8, 16],
    output_lengths: list[int] = [32, 128, 256],
    output_file: Optional[str] = None,
):
    """Run comprehensive benchmark suite."""
    
    print(f"\n{'='*60}")
    print("Light-MoE End-to-End Benchmark Suite")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output lengths: {output_lengths}")
    print(f"GPUs: {torch.cuda.device_count()}")
    print(f"{'='*60}\n")
    
    # Load model
    try:
        from light_moe import LightMoEEngine
        
        engine = LightMoEEngine(
            model_path=model_path,
            device="cuda",
            dtype="float16",
        )
        model = engine.get_model()
        tokenizer = engine._tokenizer
        
    except Exception as e:
        print(f"Could not load Light-MoE model: {e}")
        print("Falling back to HuggingFace model for benchmark structure demo")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading model...")
        # For demo, use a small model
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    # Run benchmarks
    for batch_size in batch_sizes:
        for output_length in output_lengths:
            print(f"\nBenchmarking: batch_size={batch_size}, output_length={output_length}")
            
            prompts = ["Hello, I am a large language model and"] * batch_size
            
            try:
                result = benchmark_inference(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    output_length=output_length,
                )
                results.append(asdict(result))
                
                print(f"  Throughput: {result.throughput_tokens_per_s:.1f} tokens/s")
                print(f"  TTFT: {result.time_to_first_token_ms:.1f} ms")
                print(f"  TPOT: {result.time_per_output_token_ms:.2f} ms")
                print(f"  Peak memory: {result.peak_memory_gb:.2f} GB")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    if results:
        best_throughput = max(r['throughput_tokens_per_s'] for r in results)
        best_latency = min(r['time_per_output_token_ms'] for r in results)
        
        print(f"Peak throughput: {best_throughput:.1f} tokens/s")
        print(f"Best TPOT: {best_latency:.2f} ms")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Inference Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    run_benchmark_suite(
        model_path=args.model,
        batch_sizes=args.batch_sizes,
        output_lengths=args.output_lengths,
        output_file=args.output,
    )
