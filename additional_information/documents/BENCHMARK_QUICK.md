# Performance Benchmark

## 빠른 실행

```bash
# 기본 실행 (JSON 결과 저장, 요약 출력)
python benchmark.py

# 상세 출력
python benchmark.py --verbose

# 커스텀 설정
python benchmark.py --batch-size 4 --seq-length 1024 --benchmark-steps 100
```

## 측정 항목

### 1. Wall Clock Time ⏱️
- **Training**: Forward + Backward pass 시간
- **Inference**: Forward pass만
- 통계: mean, median, std, p95, p99

### 2. Memory Consumption 💾
- **Peak Memory**: 최대 메모리 사용량
- **Current Memory**: 현재 할당된 메모리
- **Reserved Memory**: GPU에 예약된 전체 메모리

### 3. Generation Latency 🚀
- **First Token Latency**: Prefill 시간 (전체 프롬프트 처리)
- **Per-token Latency**: Decode 시간 (토큰 하나 생성)
- **Throughput**: tokens/sec

## 결과 예시

```
📊 Training Performance
  Speedup: 1.45x
  Time Reduction: 31.2%
  Memory Reduction: 245.3 MB (18.5%)

📊 Inference Performance
  Speedup: 1.62x
  Time Reduction: 38.3%
  Memory Reduction: 189.7 MB

📊 Generation Latency
  Per-token Speedup: 1.55x
  Latency Reduction: 35.5%
  First Token Speedup: 1.48x
```

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--verbose`, `-v` | False | 상세 출력 |
| `--output`, `-o` | `benchmark_results.json` | 결과 저장 경로 |
| `--batch-size` | 2 | 배치 크기 |
| `--seq-length` | 512 | 시퀀스 길이 |
| `--warmup-steps` | 10 | Warmup 스텝 수 |
| `--benchmark-steps` | 50 | 벤치마크 스텝 수 |
| `--no-compile` | False | torch.compile 비활성화 |
| `--compile-mode` | `default` | compile 모드 |

## Warmup의 중요성

**10 step warmup**을 수행하는 이유:

1. **torch.compile 최적화**
   - 첫 실행 시 CUDA graph 생성
   - Kernel fusion 최적화
   - 메모리 레이아웃 최적화

2. **Triton Kernel 최적화**
   - GPU 아키텍처별 튜닝
   - Autotuning 완료
   - Kernel cache 생성

3. **CUDA 초기화**
   - cuBLAS/cuDNN 라이브러리 초기화
   - GPU memory pool 설정

## JSON 결과 구조

```json
{
  "timestamp": "2025-10-15T...",
  "config": {
    "batch_size": 2,
    "seq_length": 512,
    "num_warmup_steps": 10,
    "use_compile": true
  },
  "default_model": {
    "training": {
      "wall_clock_time": { "mean_ms": 45.2, ... },
      "memory": { "peak_mb": 1234.5, ... },
      "throughput": { "tokens_per_sec": 22691 }
    },
    "inference": { ... },
    "generation": {
      "first_token_latency": { ... },
      "per_token_latency": { ... }
    }
  },
  "example_model": { ... },
  "comparison": {
    "training": { "speedup": 1.45, ... },
    "inference": { "speedup": 1.62, ... },
    "generation": { "speedup": 1.55, ... }
  }
}
```

## 분석 팁

```bash
# 요약만 보기
cat benchmark_results.json | jq '.comparison'

# Training speedup 확인
cat benchmark_results.json | jq '.comparison.training.speedup'

# 메모리 사용량 비교
cat benchmark_results.json | jq '{
  default: .default_model.training.memory.peak_mb,
  example: .example_model.training.memory.peak_mb
}'

# Generation latency 비교
cat benchmark_results.json | jq '{
  default: .default_model.generation.per_token_latency.mean_ms,
  example: .example_model.generation.per_token_latency.mean_ms
}'
```

## 참고사항

- 랜덤 데이터 사용 (실제 생성 불필요)
- Mixed precision (bfloat16) 사용
- CUDA synchronization으로 정확한 시간 측정
- 각 벤치마크 간 메모리 정리 수행
