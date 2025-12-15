#!/usr/bin/env python3
"""
Performance Benchmark - Default vs Example Model Comparison
===========================================================

성능 지표 측정:
1. Wall Clock Time - 훈련/추론 시간
2. Memory Consumption - GPU 메모리 사용량
3. Generation Latency - 토큰 생성 지연시간

Warmup: 10 steps (torch.compile 및 Triton kernel 최적화 고려)
"""

import sys
import torch
import torch.nn.functional as F
import transformers
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import warnings
import json
import time
from datetime import datetime
from pathlib import Path
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import models to benchmark
from models.default_model import TransformerForCausalLM, TransformerConfig
from models.example_model import ExampleTransformerForCausalLM
from models.example_config import ExampleConfig


@dataclass
class BenchmarkConfig:
    """벤치마크 설정"""
    # Model config
    vocab_size: int = 1000
    hidden_size: int = 512
    intermediate_size: int = 1376
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    
    # Benchmark config
    batch_size: int = 2
    seq_length: int = 512
    num_warmup_steps: int = 10
    num_benchmark_steps: int = 50
    
    # Generation config
    gen_batch_size: int = 1
    gen_input_length: int = 128
    gen_output_length: int = 128
    gen_num_iterations: int = 20
    
    # Compile config
    use_compile: bool = True
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GPUMemoryTracker:
    """GPU 메모리 사용량 추적"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and "cuda" in device
    
    def reset(self):
        """메모리 통계 초기화"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_current_memory(self) -> float:
        """현재 메모리 사용량 (MB)"""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device) / 1024**2
        return 0.0
    
    def get_peak_memory(self) -> float:
        """피크 메모리 사용량 (MB)"""
        if self.is_cuda:
            return torch.cuda.max_memory_allocated(self.device) / 1024**2
        return 0.0
    
    def get_reserved_memory(self) -> float:
        """예약된 메모리 (MB)"""
        if self.is_cuda:
            return torch.cuda.memory_reserved(self.device) / 1024**2
        return 0.0


class PerformanceBenchmark:
    """성능 벤치마크 메인 클래스"""
    
    def __init__(self, config: BenchmarkConfig, verbose: bool = False):
        self.config = config
        self.device = torch.device(config.device)
        self.verbose = verbose
        self.memory_tracker = GPUMemoryTracker(config.device)
        self.results: Dict = {}
        
        if self.verbose:
            print("="*80)
            print("Performance Benchmark - Default vs Example Model")
            print("="*80)
            print(f"Device: {self.device}")
            print(f"PyTorch Version: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print()
    
    def _print(self, *args, **kwargs):
        """조건부 print"""
        if self.verbose:
            print(*args, **kwargs)
    
    def _create_default_model(self) -> TransformerForCausalLM:
        """Default model 생성"""
        config = TransformerConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            max_position_embeddings=self.config.max_position_embeddings,
        )
        model = TransformerForCausalLM(config).to(self.device)
        
        if self.config.use_compile:
            self._print(f"  Compiling default model (mode={self.config.compile_mode})...")
            model = torch.compile(model, mode=self.config.compile_mode)
        
        return model
    
    def _create_example_model(self) -> ExampleTransformerForCausalLM:
        """Example model 생성"""
        config = ExampleConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            max_position_embeddings=self.config.max_position_embeddings,
        )
        model = ExampleTransformerForCausalLM(config).to(self.device)
        
        if self.config.use_compile:
            self._print(f"  Compiling example model (mode={self.config.compile_mode})...")
            model = torch.compile(model, mode=self.config.compile_mode)
        
        return model
    
    def _warmup(self, model, num_steps: int = 10):
        """
        Warmup phase for torch.compile and Triton kernels
        
        torch.compile의 최적화와 Triton kernel의 GPU 최적화를 위해
        실제 벤치마크 전에 warmup 실행
        """
        self._print(f"  Warming up for {num_steps} steps...")
        model.train()
        
        for i in range(num_steps):
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (self.config.batch_size, self.config.seq_length),
                device=self.device
            )
            labels = input_ids.clone()
            
            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            
            # Backward pass (for training warmup)
            loss.backward()
            
            # Clear gradients
            model.zero_grad(set_to_none=True)
            
            if self.verbose and (i + 1) % 5 == 0:
                self._print(f"    Warmup step {i+1}/{num_steps}")
        
        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self._print(f"  ✓ Warmup completed")
    
    def benchmark_training(self, model, model_name: str) -> Dict:
        """훈련 성능 벤치마크"""
        self._print(f"\n[Benchmark] Training Performance - {model_name}")
        
        # Reset memory tracker
        self.memory_tracker.reset()
        
        # Warmup
        self._warmup(model, self.config.num_warmup_steps)
        
        # Reset memory after warmup
        self.memory_tracker.reset()
        
        model.train()
        times = []
        memory_snapshots = []
        
        self._print(f"  Running {self.config.num_benchmark_steps} training steps...")
        
        for i in range(self.config.num_benchmark_steps):
            # Generate random data
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (self.config.batch_size, self.config.seq_length),
                device=self.device
            )
            labels = input_ids.clone()
            
            # Measure time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Forward + Backward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            
            loss.backward()
            model.zero_grad(set_to_none=True)
            
            # Synchronize and measure
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            step_time = (end_time - start_time) * 1000  # ms
            times.append(step_time)
            
            # Memory snapshot
            memory_snapshots.append({
                'current': self.memory_tracker.get_current_memory(),
                'peak': self.memory_tracker.get_peak_memory(),
            })
            
            if self.verbose and (i + 1) % 10 == 0:
                self._print(f"    Step {i+1}/{self.config.num_benchmark_steps}: "
                          f"{step_time:.2f}ms")
        
        # Calculate statistics
        times = np.array(times)
        results = {
            'wall_clock_time': {
                'mean_ms': float(np.mean(times)),
                'std_ms': float(np.std(times)),
                'median_ms': float(np.median(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'p95_ms': float(np.percentile(times, 95)),
                'p99_ms': float(np.percentile(times, 99)),
            },
            'memory': {
                'peak_mb': self.memory_tracker.get_peak_memory(),
                'final_mb': self.memory_tracker.get_current_memory(),
                'reserved_mb': self.memory_tracker.get_reserved_memory(),
            },
            'throughput': {
                'samples_per_sec': 1000.0 / np.mean(times) * self.config.batch_size,
                'tokens_per_sec': 1000.0 / np.mean(times) * self.config.batch_size * self.config.seq_length,
            }
        }
        
        self._print(f"  ✓ Training benchmark completed")
        self._print(f"    Mean time: {results['wall_clock_time']['mean_ms']:.2f}ms")
        self._print(f"    Peak memory: {results['memory']['peak_mb']:.2f}MB")
        self._print(f"    Throughput: {results['throughput']['tokens_per_sec']:.0f} tokens/sec")
        
        return results
    
    def benchmark_inference(self, model, model_name: str) -> Dict:
        """추론 성능 벤치마크"""
        self._print(f"\n[Benchmark] Inference Performance - {model_name}")
        
        # Reset memory tracker
        self.memory_tracker.reset()
        
        # Warmup (inference mode)
        self._print(f"  Warming up for {self.config.num_warmup_steps} steps...")
        model.eval()
        
        for i in range(self.config.num_warmup_steps):
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (self.config.batch_size, self.config.seq_length),
                device=self.device
            )
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self._print(f"  ✓ Warmup completed")
        
        # Reset memory after warmup
        self.memory_tracker.reset()
        
        times = []
        
        self._print(f"  Running {self.config.num_benchmark_steps} inference steps...")
        
        for i in range(self.config.num_benchmark_steps):
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (self.config.batch_size, self.config.seq_length),
                device=self.device
            )
            
            # Measure time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            step_time = (end_time - start_time) * 1000  # ms
            times.append(step_time)
            
            if self.verbose and (i + 1) % 10 == 0:
                self._print(f"    Step {i+1}/{self.config.num_benchmark_steps}: "
                          f"{step_time:.2f}ms")
        
        # Calculate statistics
        times = np.array(times)
        results = {
            'wall_clock_time': {
                'mean_ms': float(np.mean(times)),
                'std_ms': float(np.std(times)),
                'median_ms': float(np.median(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'p95_ms': float(np.percentile(times, 95)),
                'p99_ms': float(np.percentile(times, 99)),
            },
            'memory': {
                'peak_mb': self.memory_tracker.get_peak_memory(),
                'final_mb': self.memory_tracker.get_current_memory(),
                'reserved_mb': self.memory_tracker.get_reserved_memory(),
            },
            'throughput': {
                'samples_per_sec': 1000.0 / np.mean(times) * self.config.batch_size,
                'tokens_per_sec': 1000.0 / np.mean(times) * self.config.batch_size * self.config.seq_length,
            }
        }
        
        self._print(f"  ✓ Inference benchmark completed")
        self._print(f"    Mean time: {results['wall_clock_time']['mean_ms']:.2f}ms")
        self._print(f"    Peak memory: {results['memory']['peak_mb']:.2f}MB")
        self._print(f"    Throughput: {results['throughput']['tokens_per_sec']:.0f} tokens/sec")
        
        return results
    
    def benchmark_generation(self, model, model_name: str) -> Dict:
        """
        생성 지연시간 벤치마크
        
        Autoregressive generation의 latency 측정:
        - First token latency (prefill)
        - Per-token latency (decode)
        """
        self._print(f"\n[Benchmark] Generation Latency - {model_name}")
        
        # Reset memory tracker
        self.memory_tracker.reset()
        
        # Warmup
        self._print(f"  Warming up for {self.config.num_warmup_steps} iterations...")
        model.eval()
        
        for i in range(self.config.num_warmup_steps):
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (self.config.gen_batch_size, self.config.gen_input_length),
                device=self.device
            )
            
            # Simulate generation
            past_key_values = None
            for _ in range(min(10, self.config.gen_output_length)):
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                past_key_values = outputs.past_key_values
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self._print(f"  ✓ Warmup completed")
        
        # Reset memory after warmup
        self.memory_tracker.reset()
        
        first_token_latencies = []
        per_token_latencies = []
        total_latencies = []
        
        self._print(f"  Running {self.config.gen_num_iterations} generation iterations...")
        
        for i in range(self.config.gen_num_iterations):
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (self.config.gen_batch_size, self.config.gen_input_length),
                device=self.device
            )
            
            # Measure total generation time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_start = time.perf_counter()
            
            past_key_values = None
            token_times = []
            
            for j in range(self.config.gen_output_length):
                # Prefill (first token) or decode (subsequent tokens)
                curr_input = input_ids if past_key_values is None else input_ids[:, -1:]
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                token_start = time.perf_counter()
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=curr_input,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                token_end = time.perf_counter()
                
                token_time = (token_end - token_start) * 1000  # ms
                token_times.append(token_time)
                
                # Get next token (random for benchmark)
                next_token = torch.randint(
                    0, self.config.vocab_size,
                    (self.config.gen_batch_size, 1),
                    device=self.device
                )
                input_ids = torch.cat([input_ids, next_token], dim=1)
                past_key_values = outputs.past_key_values
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_end = time.perf_counter()
            
            total_time = (total_end - total_start) * 1000  # ms
            
            # First token is prefill, rest are decode
            first_token_latencies.append(token_times[0])
            if len(token_times) > 1:
                per_token_latencies.extend(token_times[1:])
            total_latencies.append(total_time)
            
            if self.verbose and (i + 1) % 5 == 0:
                self._print(f"    Iteration {i+1}/{self.config.gen_num_iterations}: "
                          f"first={token_times[0]:.2f}ms, "
                          f"avg_decode={np.mean(token_times[1:]):.2f}ms")
        
        # Calculate statistics
        first_token_latencies = np.array(first_token_latencies)
        per_token_latencies = np.array(per_token_latencies)
        total_latencies = np.array(total_latencies)
        
        results = {
            'first_token_latency': {
                'mean_ms': float(np.mean(first_token_latencies)),
                'std_ms': float(np.std(first_token_latencies)),
                'median_ms': float(np.median(first_token_latencies)),
                'p95_ms': float(np.percentile(first_token_latencies, 95)),
                'p99_ms': float(np.percentile(first_token_latencies, 99)),
            },
            'per_token_latency': {
                'mean_ms': float(np.mean(per_token_latencies)),
                'std_ms': float(np.std(per_token_latencies)),
                'median_ms': float(np.median(per_token_latencies)),
                'p95_ms': float(np.percentile(per_token_latencies, 95)),
                'p99_ms': float(np.percentile(per_token_latencies, 99)),
            },
            'total_generation': {
                'mean_ms': float(np.mean(total_latencies)),
                'std_ms': float(np.std(total_latencies)),
                'output_tokens': self.config.gen_output_length,
            },
            'memory': {
                'peak_mb': self.memory_tracker.get_peak_memory(),
                'final_mb': self.memory_tracker.get_current_memory(),
            },
            'throughput': {
                'tokens_per_sec': 1000.0 / np.mean(per_token_latencies),
            }
        }
        
        self._print(f"  ✓ Generation benchmark completed")
        self._print(f"    First token latency: {results['first_token_latency']['mean_ms']:.2f}ms")
        self._print(f"    Per-token latency: {results['per_token_latency']['mean_ms']:.2f}ms")
        self._print(f"    Throughput: {results['throughput']['tokens_per_sec']:.1f} tokens/sec")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """전체 벤치마크 실행"""
        self._print("\n" + "="*80)
        self._print("Starting Full Performance Benchmark")
        self._print("="*80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'batch_size': self.config.batch_size,
                'seq_length': self.config.seq_length,
                'vocab_size': self.config.vocab_size,
                'hidden_size': self.config.hidden_size,
                'num_hidden_layers': self.config.num_hidden_layers,
                'num_warmup_steps': self.config.num_warmup_steps,
                'num_benchmark_steps': self.config.num_benchmark_steps,
                'use_compile': self.config.use_compile,
                'compile_mode': self.config.compile_mode,
                'device': str(self.device),
            },
            'default_model': {},
            'example_model': {},
            'comparison': {},
        }
        
        # Benchmark Default Model
        self._print("\n" + "="*80)
        self._print("Benchmarking Default Model")
        self._print("="*80)
        
        default_model = self._create_default_model()
        
        results['default_model']['training'] = self.benchmark_training(
            default_model, "Default Model"
        )
        
        # Clean up before inference
        torch.cuda.empty_cache()
        gc.collect()
        
        results['default_model']['inference'] = self.benchmark_inference(
            default_model, "Default Model"
        )
        
        # Clean up before generation
        torch.cuda.empty_cache()
        gc.collect()
        
        results['default_model']['generation'] = self.benchmark_generation(
            default_model, "Default Model"
        )
        
        # Clean up default model
        del default_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark Example Model
        self._print("\n" + "="*80)
        self._print("Benchmarking Example Model")
        self._print("="*80)
        
        example_model = self._create_example_model()
        
        results['example_model']['training'] = self.benchmark_training(
            example_model, "Example Model"
        )
        
        # Clean up before inference
        torch.cuda.empty_cache()
        gc.collect()
        
        results['example_model']['inference'] = self.benchmark_inference(
            example_model, "Example Model"
        )
        
        # Clean up before generation
        torch.cuda.empty_cache()
        gc.collect()
        
        results['example_model']['generation'] = self.benchmark_generation(
            example_model, "Example Model"
        )
        
        # Clean up example model
        del example_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Calculate improvements
        results['comparison'] = self._calculate_improvements(
            results['default_model'],
            results['example_model']
        )
        
        self.results = results
        return results
    
    def _calculate_improvements(self, default_results: Dict, example_results: Dict) -> Dict:
        """성능 개선율 계산"""
        comparison = {}
        
        # Training comparison
        default_train_time = default_results['training']['wall_clock_time']['mean_ms']
        example_train_time = example_results['training']['wall_clock_time']['mean_ms']
        
        comparison['training'] = {
            'speedup': default_train_time / example_train_time,
            'time_reduction_percent': (1 - example_train_time / default_train_time) * 100,
            'memory_reduction_mb': (
                default_results['training']['memory']['peak_mb'] -
                example_results['training']['memory']['peak_mb']
            ),
            'memory_reduction_percent': (
                1 - example_results['training']['memory']['peak_mb'] /
                default_results['training']['memory']['peak_mb']
            ) * 100,
        }
        
        # Inference comparison
        default_inf_time = default_results['inference']['wall_clock_time']['mean_ms']
        example_inf_time = example_results['inference']['wall_clock_time']['mean_ms']
        
        comparison['inference'] = {
            'speedup': default_inf_time / example_inf_time,
            'time_reduction_percent': (1 - example_inf_time / default_inf_time) * 100,
            'memory_reduction_mb': (
                default_results['inference']['memory']['peak_mb'] -
                example_results['inference']['memory']['peak_mb']
            ),
        }
        
        # Generation comparison
        default_gen_latency = default_results['generation']['per_token_latency']['mean_ms']
        example_gen_latency = example_results['generation']['per_token_latency']['mean_ms']
        
        comparison['generation'] = {
            'speedup': default_gen_latency / example_gen_latency,
            'latency_reduction_percent': (1 - example_gen_latency / default_gen_latency) * 100,
            'first_token_speedup': (
                default_results['generation']['first_token_latency']['mean_ms'] /
                example_results['generation']['first_token_latency']['mean_ms']
            ),
        }
        
        return comparison
    
    def print_summary(self):
        """결과 요약 출력"""
        if not self.results:
            print("No results to display. Run benchmark first.")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        comp = self.results['comparison']
        
        print("\n📊 Training Performance")
        print(f"  Speedup: {comp['training']['speedup']:.2f}x")
        print(f"  Time Reduction: {comp['training']['time_reduction_percent']:.1f}%")
        print(f"  Memory Reduction: {comp['training']['memory_reduction_mb']:.1f} MB "
              f"({comp['training']['memory_reduction_percent']:.1f}%)")
        
        print("\n📊 Inference Performance")
        print(f"  Speedup: {comp['inference']['speedup']:.2f}x")
        print(f"  Time Reduction: {comp['inference']['time_reduction_percent']:.1f}%")
        print(f"  Memory Reduction: {comp['inference']['memory_reduction_mb']:.1f} MB")
        
        print("\n📊 Generation Latency")
        print(f"  Per-token Speedup: {comp['generation']['speedup']:.2f}x")
        print(f"  Latency Reduction: {comp['generation']['latency_reduction_percent']:.1f}%")
        print(f"  First Token Speedup: {comp['generation']['first_token_speedup']:.2f}x")
        
        print("\n" + "="*80)
    
    def save_results_json(self, output_path: str = "benchmark_results.json"):
        """결과를 JSON 파일로 저장"""
        if not self.results:
            print("No results to save. Run benchmark first.")
            return None
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n📄 Results saved to: {output_path}")
        
        return output_path


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance benchmark for default vs example model")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed output to console")
    parser.add_argument("--output", "-o", type=str,
                       default="benchmark_results.json",
                       help="Output JSON file path (default: benchmark_results.json)")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training/inference (default: 2)")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Sequence length (default: 512)")
    parser.add_argument("--warmup-steps", type=int, default=10,
                       help="Number of warmup steps (default: 10)")
    parser.add_argument("--benchmark-steps", type=int, default=50,
                       help="Number of benchmark steps (default: 50)")
    parser.add_argument("--no-compile", action="store_true",
                       help="Disable torch.compile")
    parser.add_argument("--compile-mode", type=str, default="default",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode (default: default)")
    
    args = parser.parse_args()
    
    # Benchmark configuration
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_warmup_steps=args.warmup_steps,
        num_benchmark_steps=args.benchmark_steps,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode,
    )
    
    # Run benchmark
    benchmark = PerformanceBenchmark(config, verbose=args.verbose)
    results = benchmark.run_full_benchmark()
    
    # Save results
    output_path = benchmark.save_results_json(args.output)
    
    # Print summary
    if not args.verbose:
        benchmark.print_summary()
        print(f"\n✅ Benchmark completed - Results saved to {output_path}")
    else:
        benchmark.print_summary()
    
    # Exit
    sys.exit(0)


if __name__ == "__main__":
    main()
