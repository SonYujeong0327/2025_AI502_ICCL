# Sanity Check - Default Model Verification

## 개요

`sanity_check.py`는 제출된 `default_model`이 수학적으로 정확한지 검증하는 자동화된 테스트 스크립트입니다.

**⚠️ 중요**: 이 테스트를 통과하지 못하면 **DESK REJECT** 됩니다.

## 실행 방법

```bash
# 기본 실행 (JSON 파일로 결과 저장, 최소 출력)
python sanity_check.py

# Verbose 모드 (상세한 콘솔 출력)
python sanity_check.py --verbose
python sanity_check.py -v

# 커스텀 출력 파일 경로
python sanity_check.py --output results/my_check.json
python sanity_check.py -o results/my_check.json

# 가상환경에서 실행
source .venv/bin/activate
python sanity_check.py
```

### 출력 옵션

- **기본 모드** (권장): 최소한의 콘솔 출력, 결과를 JSON 파일로 저장
  ```bash
  python sanity_check.py
  # 출력: ✅ All 11 tests passed - Results saved to sanity_check_results.json
  ```

- **Verbose 모드**: 각 테스트의 상세 정보를 콘솔에 출력
  ```bash
  python sanity_check.py --verbose
  ```

- **커스텀 출력 파일**: 결과 저장 위치 지정
  ```bash
  python sanity_check.py -o trainer_output/sanity_check.json
  ```

## 검증 항목

### 1. Layer Structure Validation
- 모델의 전체 구조가 올바른지 검증
- Embedding, Decoder layers, LM head 등의 존재 여부
- Attention head 수, KV head 수, hidden dimension 등의 설정 확인

### 2. RMSNorm Mathematical Correctness
- RMSNorm의 수학적 정확성 검증
- 공식: `x * rsqrt(mean(x^2) + eps) * weight`
- Liger Kernel 구현과의 일관성

### 3. RotaryEmbedding Correctness
- Rotary Position Embedding의 출력 shape 검증
- cos/sin 값의 범위 확인
- NaN/Inf 값 체크

### 4. Multi-Head Attention Forward
- Attention 메커니즘의 forward pass 검증
- Query, Key, Value projection 정확성
- Grouped-Query Attention (GQA) 동작 확인
- 출력 값의 범위 및 안정성

### 5. SwiGLU MLP Forward
- SwiGLU activation을 사용한 MLP 검증
- Gate projection과 Up projection의 정확성
- Liger Kernel의 fused implementation 사용 확인

### 6. TransformerDecoderLayer Forward
- 전체 decoder layer의 forward pass
- Pre/Post normalization 적용 확인
- Residual connection 동작

### 7. Full Model Forward Pass
- 전체 모델의 end-to-end forward pass
- Input embedding → Decoder layers → LM head
- Logits 출력의 shape 및 값 범위

### 8. Loss Calculation Correctness
- Cross-entropy loss 계산의 정확성
- Fused linear cross-entropy 구현 검증
- Loss 값의 합리적인 범위 확인 (3.0 ~ 15.0)

### 9. KV Cache Correctness
- Key-Value cache의 동작 정확성
- Full context vs Cached context 비교
- Incremental generation 수치 안정성
- Cache 사용 시와 미사용 시 출력 일치성

### 10. Gradient Flow
- 모든 학습 가능한 파라미터로 gradient 전달 확인
- Backward pass의 정확성
- Gradient vanishing/exploding 체크

### 11. Numerical Stability
- 여러 번의 inference에서 수치 안정성
- Extreme input에 대한 robustness
- NaN/Inf 발생 여부

## 테스트 설정

기본 테스트 설정 (`TestConfig`):

```python
batch_size: 2
seq_length: 32
vocab_size: 1000
hidden_size: 512
intermediate_size: 1376
num_hidden_layers: 4
num_attention_heads: 8
num_key_value_heads: 2
max_position_embeddings: 2048
rms_norm_eps: 1e-6
rope_theta: 10000.0
attention_dropout: 0.0
```

허용 오차:
- `rtol`: 1e-3 (0.1%)
- `atol`: 1e-4

## 출력 형식

### JSON 파일 (기본)

`sanity_check_results.json` 파일이 생성되며, 다음 정보를 포함합니다:

```json
{
  "timestamp": "2025-10-15T01:55:29.328801",
  "summary": {
    "total_tests": 11,
    "passed": 11,
    "failed": 0,
    "success_rate": "100.0%",
    "overall_status": "PASSED"
  },
  "environment": {
    "device": "cuda",
    "pytorch_version": "2.8.0+cu129",
    "transformers_version": "4.57.0",
    "cuda_available": true
  },
  "test_config": {
    "batch_size": 2,
    "seq_length": 32,
    "vocab_size": 1000,
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "rtol": 0.001,
    "atol": 0.0001
  },
  "test_results": {
    "Layer Structure Validation": {
      "passed": true,
      "details": {}
    },
    ...
  },
  "errors": []
}
```

### 콘솔 출력

**기본 모드** (최소 출력):
```
✅ All 11 tests passed - Results saved to sanity_check_results.json
```

Exit code: 0

**Verbose 모드** (--verbose):
```
================================================================================
Default Model Sanity Check - Mathematical Correctness Verification
================================================================================
Device: cuda
PyTorch Version: 2.8.0+cu129
Transformers Version: 4.57.0

Running Tests...
--------------------------------------------------------------------------------

[TEST] Layer Structure Validation
  ✓ All 35 structure checks passed
✅ PASSED: Layer Structure Validation
...
================================================================================
🎉 ALL TESTS PASSED - Model is mathematically correct!
================================================================================
```

### 실패 시 (Verbose)

```
================================================================================
🎉 ALL TESTS PASSED - Model is mathematically correct!
================================================================================
```

Exit code: 0

### 실패 시 (Verbose)

```
================================================================================
⚠️  SOME TESTS FAILED - DESK REJECT
================================================================================
Total: X/11 tests passed

Detailed Errors:
  - [Test Name]: [Error Message]
```

**기본 모드**:
```
❌ X/11 tests failed - Results saved to sanity_check_results.json
```

Exit code: 1

## 일반적인 실패 원인

### 1. Layer Structure 실패
- `default_model/` 파일을 수정한 경우
- 필수 레이어가 누락된 경우
- 레이어 파라미터가 올바르지 않은 경우

### 2. Forward Pass 실패
- Shape mismatch (차원 불일치)
- NaN/Inf 발생
- Activation function 오류

### 3. Loss Calculation 실패
- Fused linear cross-entropy 구현 오류
- Label shifting 문제
- Loss 값이 비정상적인 범위

### 4. KV Cache 실패
- Cache update 로직 오류
- Position encoding 처리 문제
- Cache index 계산 오류

### 5. Gradient Flow 실패
- Backward pass가 막힌 경우
- Detached tensor 사용
- In-place operation 문제

## 디버깅 팁

### 1. JSON 결과 분석
```bash
# JSON 파일 읽기
python -m json.tool sanity_check_results.json

# jq 사용 (더 예쁜 출력)
jq . sanity_check_results.json

# 실패한 테스트만 보기
jq '.test_results | to_entries | map(select(.value.passed == false))' sanity_check_results.json
```

### 2. 특정 테스트만 실행
```python
# sanity_check.py 수정
tests = [
    ("Loss Calculation Correctness", self.test_loss_calculation),
]
```

### 2. 특정 테스트만 실행
```python
# sanity_check.py 수정
tests = [
    ("Loss Calculation Correctness", self.test_loss_calculation),
]
```

### 3. Verbose 모드로 더 자세한 출력
```bash
python sanity_check.py --verbose 2>&1 | tee debug.log
```

### 4. Tolerance 조정 (임시)
```python
# TestConfig에서 조정 (제출 전 원복 필수)
rtol: float = 1e-2  # 더 관대한 tolerance
atol: float = 1e-3
```

## 확장 가능성

### Optional MLP 변형

MLP 아키텍처를 변경한 경우 (예: GeGLU 사용):

1. `sanity_check.py`의 `test_mlp_forward` 수정
2. 보고서에 변경 사항 명시
3. TA에게 연락하여 example sanity check 요청

### 추가 테스트 작성

프로젝트 특정 최적화를 검증하기 위한 추가 테스트:

```python
def test_example_optimization(self) -> bool:
    """커스텀 최적화 기능 검증"""
    # 구현...
    return passed
```

## CI/CD 통합

GitHub Actions 예시:

```yaml
name: Sanity Check
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.13'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run sanity check
        run: python sanity_check.py
```

## 제출 전 체크리스트

- [ ] `python sanity_check.py` 실행하여 모든 테스트 통과 확인
- [ ] `sanity_check_results.json` 파일 생성 확인
- [ ] JSON에서 `"overall_status": "PASSED"` 확인
- [ ] `default_model/` 디렉토리의 파일을 수정하지 않았는지 확인
- [ ] Exit code가 0인지 확인: `echo $?`

## 자동화 스크립트

### 제출 전 검증
```bash
#!/bin/bash
python sanity_check.py
if [ $? -eq 0 ]; then
    echo "✅ 제출 가능!"
    cat sanity_check_results.json | jq '.summary'
else
    echo "❌ DESK REJECT 위험!"
    cat sanity_check_results.json | jq '.errors'
    exit 1
fi
```

## 문의

테스트 실패 원인을 알 수 없거나 false positive라고 판단되는 경우:

1. 출력 로그 전체를 캡처
2. 재현 가능한 최소 예제 작성
3. TA에게 문의 (단, trivial한 코드 질문은 답변하지 않음)

## 참고 자료

- Liger Kernel: https://github.com/linkedin/Liger-Kernel
- Transformers: https://github.com/huggingface/transformers
- Flash Attention: https://github.com/Dao-AILab/flash-attention
