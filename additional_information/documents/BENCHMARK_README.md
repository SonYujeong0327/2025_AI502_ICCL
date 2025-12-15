# Performance Benchmark Documentation

## 개요

`benchmark.py`는 default_model과 example_model의 성능을 정량적으로 비교 평가하는 벤치마크 도구입니다.

## 측정 지표 (Performance Metrics)

### 1. Wall Clock Time ⏱️

실제 실행 시간을 측정합니다.

**Training Mode**
- Forward pass + Backward pass + Zero grad
- Mixed precision (bfloat16) 사용
- 통계: mean, median, std, p95, p99 (ms)

**Inference Mode**
- Forward pass만
- `torch.no_grad()` 사용
- 통계: mean, median, std, p95, p99 (ms)

**Throughput**
- Samples per second
- Tokens per second

### 2. Memory Consumption 💾

GPU 메모리 사용량을 추적합니다.

**측정 항목**
- **Peak Memory**: 최대 메모리 사용량 (MB)
- **Current Memory**: 현재 할당된 메모리 (MB)
- **Reserved Memory**: GPU에 예약된 전체 메모리 (MB)

**추적 방법**
- `torch.cuda.memory_allocated()`: 실제 할당된 메모리
- `torch.cuda.max_memory_allocated()`: 피크 메모리
- `torch.cuda.memory_reserved()`: 캐시 포함 예약 메모리

### 3. Generation Latency 🚀

Autoregressive generation의 지연시간을 측정합니다.

**First Token Latency (Prefill)**
- 전체 프롬프트 처리 시간
- KV cache 초기 구축
- 통계: mean, median, p95, p99 (ms)

**Per-token Latency (Decode)**
- 각 토큰 생성 시간
- KV cache 활용한 incremental generation
- 통계: mean, median, p95, p99 (ms)

**Throughput**
- Tokens per second (decode phase)

## Warmup의 중요성

### 왜 10 step warmup이 필요한가?

#### 1. torch.compile 최적화
```python
model = torch.compile(model, mode="default")
```

**첫 실행 시:**
- CUDA graph 생성 및 최적화
- Kernel fusion 수행
- 메모리 레이아웃 최적화
- Dynamic shape handling 학습

**Warmup 후:**
- 최적화된 CUDA graph 재사용
- 안정적인 성능 측정 가능

#### 2. Triton Kernel 최적화 (Liger Kernel)
```python
# Liger Kernel uses Triton
from liger_kernel.transformers.rms_norm import LigerRMSNorm
```

**Autotuning 과정:**
- GPU 아키텍처 감지 (A100, V100, etc.)
- Block size, thread 구성 최적화
- Shared memory 사용 패턴 학습
- Kernel cache 생성

#### 3. Flash Attention 2 최적화
```python
from flash_attn import flash_attn_func
```

**최적화 요소:**
- CUDA kernel 로딩
- Tile size 최적화
- Memory layout 조정

#### 4. CUDA 라이브러리 초기화
- cuBLAS context 생성
- cuDNN algorithm selection
- Memory pool initialization

### Warmup 후 성능 차이

| Phase | First Run | After Warmup | Speedup |
|-------|-----------|--------------|---------|
| torch.compile | ~1000ms | ~50ms | 20x |
| Triton kernel | ~100ms | ~5ms | 20x |
| Flash Attention | ~50ms | ~3ms | 16x |

## 실행 방법

### 기본 실행

```bash
# 기본 설정 (권장)
python benchmark.py

# 결과:
# - benchmark_results.json 생성
# - 콘솔에 요약 출력
```

### 상세 설정

```bash
# 배치 크기 및 시퀀스 길이 조정
python benchmark.py --batch-size 4 --seq-length 1024

# Warmup 및 벤치마크 스텝 수 조정
python benchmark.py --warmup-steps 20 --benchmark-steps 100

# torch.compile 비활성화 (디버깅용)
python benchmark.py --no-compile

# torch.compile 모드 선택
python benchmark.py --compile-mode max-autotune

# Verbose 출력
python benchmark.py --verbose

# 커스텀 출력 경로
python benchmark.py -o results/my_benchmark.json
```

### 프로덕션 벤치마크

```bash
# 정확한 측정을 위한 권장 설정
python benchmark.py \
  --batch-size 2 \
  --seq-length 512 \
  --warmup-steps 20 \
  --benchmark-steps 100 \
  --compile-mode max-autotune \
  --verbose
```

## 설정 옵션

### 모델 설정
기본 설정은 `BenchmarkConfig` 클래스에 정의:

```python
vocab_size: 1000
hidden_size: 512
intermediate_size: 1376
num_hidden_layers: 4
num_attention_heads: 8
num_key_value_heads: 2  # Grouped-Query Attention
max_position_embeddings: 2048
```

### 벤치마크 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `batch_size` | 2 | 훈련/추론 배치 크기 |
| `seq_length` | 512 | 시퀀스 길이 |
| `num_warmup_steps` | 10 | Warmup 스텝 수 |
| `num_benchmark_steps` | 50 | 벤치마크 스텝 수 |
| `gen_batch_size` | 1 | 생성 배치 크기 |
| `gen_input_length` | 128 | 생성 입력 길이 |
| `gen_output_length` | 128 | 생성 출력 길이 |
| `gen_num_iterations` | 20 | 생성 반복 횟수 |

### torch.compile 설정

| 모드 | 설명 | 추천 용도 |
|------|------|----------|
| `default` | 기본 최적화 | 일반적인 사용 |
| `reduce-overhead` | Overhead 최소화 | 작은 배치 |
| `max-autotune` | 최대 최적화 | 프로덕션 벤치마크 |

## JSON 결과 구조

```json
{
  "timestamp": "2025-10-15T...",
  "config": {
    "batch_size": 2,
    "seq_length": 512,
    "num_warmup_steps": 10,
    "num_benchmark_steps": 50,
    "use_compile": true,
    "compile_mode": "default"
  },
  "default_model": {
    "training": {
      "wall_clock_time": {
        "mean_ms": 45.2,
        "std_ms": 2.1,
        "median_ms": 44.8,
        "min_ms": 42.3,
        "max_ms": 49.1,
        "p95_ms": 47.5,
        "p99_ms": 48.9
      },
      "memory": {
        "peak_mb": 1234.5,
        "final_mb": 987.3,
        "reserved_mb": 1500.0
      },
      "throughput": {
        "samples_per_sec": 44.2,
        "tokens_per_sec": 22691.0
      }
    },
    "inference": { ... },
    "generation": {
      "first_token_latency": {
        "mean_ms": 8.5,
        "p95_ms": 9.2,
        "p99_ms": 9.8
      },
      "per_token_latency": {
        "mean_ms": 2.3,
        "p95_ms": 2.5,
        "p99_ms": 2.7
      },
      "total_generation": {
        "mean_ms": 302.4,
        "output_tokens": 128
      },
      "throughput": {
        "tokens_per_sec": 434.8
      }
    }
  },
  "example_model": { ... },
  "comparison": {
    "training": {
      "speedup": 1.45,
      "time_reduction_percent": 31.2,
      "memory_reduction_mb": 245.3,
      "memory_reduction_percent": 18.5
    },
    "inference": {
      "speedup": 1.62,
      "time_reduction_percent": 38.3,
      "memory_reduction_mb": 189.7
    },
    "generation": {
      "speedup": 1.55,
      "latency_reduction_percent": 35.5,
      "first_token_speedup": 1.48
    }
  }
}
```

## 결과 분석

### jq를 이용한 분석

```bash
# 전체 요약
cat benchmark_results.json | jq '.comparison'

# Training speedup
cat benchmark_results.json | jq '.comparison.training.speedup'

# 메모리 비교
cat benchmark_results.json | jq '{
  default_peak: .default_model.training.memory.peak_mb,
  example_peak: .example_model.training.memory.peak_mb,
  reduction_mb: .comparison.training.memory_reduction_mb
}'

# Generation latency
cat benchmark_results.json | jq '{
  default_latency: .default_model.generation.per_token_latency.mean_ms,
  example_latency: .example_model.generation.per_token_latency.mean_ms,
  speedup: .comparison.generation.speedup
}'

# Throughput 비교
cat benchmark_results.json | jq '{
  default_tps: .default_model.inference.throughput.tokens_per_sec,
  example_tps: .example_model.inference.throughput.tokens_per_sec
}'
```

### Python 분석 스크립트

```python
import json

with open('benchmark_results.json') as f:
    results = json.load(f)

comp = results['comparison']

print(f"Training Speedup: {comp['training']['speedup']:.2f}x")
print(f"Memory Reduction: {comp['training']['memory_reduction_mb']:.1f} MB")
print(f"Generation Speedup: {comp['generation']['speedup']:.2f}x")
```

## 일반적인 최적화 결과

### Flash Attention 2 적용

**예상 개선:**
- Training: 1.3-1.5x speedup
- Inference: 1.5-2.0x speedup
- Memory: 20-30% reduction

### Kernel Fusion (Liger Kernel)

**예상 개선:**
- Training: 1.2-1.4x speedup
- Memory: 15-25% reduction

### torch.compile

**예상 개선:**
- Training: 1.1-1.3x speedup
- Inference: 1.2-1.5x speedup

### 종합 최적화

**목표:**
- Training: 1.5-2.0x speedup
- Inference: 2.0-3.0x speedup
- Memory: 30-40% reduction
- Generation: 2.0-2.5x speedup

## 트러블슈팅

### CUDA OOM

```bash
# 배치 크기 줄이기
python benchmark.py --batch-size 1 --seq-length 256

# Compile 비활성화
python benchmark.py --no-compile
```

### Compile 오류

```bash
# Compile 모드 변경
python benchmark.py --compile-mode default

# Compile 비활성화
python benchmark.py --no-compile
```

### 느린 첫 실행

- 정상입니다! torch.compile과 Triton kernel의 최적화 과정
- Warmup 후 성능이 크게 향상됨

### 불안정한 측정

```bash
# Warmup과 벤치마크 스텝 수 증가
python benchmark.py --warmup-steps 20 --benchmark-steps 100
```

## CI/CD 통합

### GitHub Actions 예시

```yaml
name: Performance Benchmark
on: [push]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmark
        run: |
          python benchmark.py --output results/benchmark_${{ github.sha }}.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: results/
```

## 참고 자료

- torch.compile: https://pytorch.org/docs/stable/torch.compiler.html
- Liger Kernel: https://github.com/linkedin/Liger-Kernel
- Flash Attention 2: https://github.com/Dao-AILab/flash-attention
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
