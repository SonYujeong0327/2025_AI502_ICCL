# Sanity Check Quick Reference

## 빠른 실행

```bash
# 기본 (JSON 결과 저장, 최소 출력)
python sanity_check.py

# 상세 출력
python sanity_check.py --verbose

# 커스텀 출력 파일
python sanity_check.py -o results/check.json
```

## 예상 결과

✅ **성공** (Exit code: 0)
```
✅ All 11 tests passed - Results saved to sanity_check_results.json
```

❌ **실패** (Exit code: 1)
```
❌ X/11 tests failed - Results saved to sanity_check_results.json
```

## JSON 결과 파일

`sanity_check_results.json`:
```json
{
  "summary": {
    "total_tests": 11,
    "passed": 11,
    "overall_status": "PASSED"
  },
  "test_results": { ... },
  "errors": []
}
```

## 11가지 테스트

| # | 테스트 항목 | 검증 내용 |
|---|------------|----------|
| 1 | Layer Structure | 모델 구조, 레이어 존재 여부 |
| 2 | RMSNorm | 수학적 정확성 (Liger Kernel) |
| 3 | RotaryEmbedding | RoPE 구현, shape 확인 |
| 4 | Attention Forward | Multi-Head Attention 동작 |
| 5 | MLP Forward | SwiGLU MLP 동작 |
| 6 | Decoder Layer | 전체 decoder layer |
| 7 | Full Model | End-to-end forward pass |
| 8 | Loss Calculation | Fused cross-entropy loss |
| 9 | KV Cache | Cache 정확성, incremental gen |
| 10 | Gradient Flow | Backward pass, 모든 파라미터 |
| 11 | Numerical Stability | NaN/Inf 체크, 안정성 |

## 실패 시 대응

### 1. default_model/ 수정 확인
```bash
git diff default_model/
```
**절대 수정 금지!**

### 2. 로그 분석
- Shape mismatch → dimension 확인
- NaN/Inf → 초기화, normalization 문제
- Gradient flow 실패 → detached tensor 확인

### 3. 테스트 재실행
```bash
python sanity_check.py 2>&1 | tee sanity_check.log
```

## 주요 허용 오차

- `rtol`: 1e-3 (0.1% 상대 오차)
- `atol`: 1e-4 (절대 오차)
- KV Cache: 1% relative error
- Incremental gen: 10% relative error

## 문제 해결 순서

1. ✅ `python sanity_check.py` 실행
2. ❌ 실패 시 → JSON 파일 확인
   ```bash
   cat sanity_check_results.json | jq '.errors'
   ```
3. 🔍 원인 파악:
   - default_model 수정? → 되돌리기
   - Import 오류? → layers.py에서 가져오기
   - Shape 오류? → dimension 재확인
4. 🔧 수정 후 재실행
5. 🎉 통과 시 → 제출 가능

## 빠른 검증

```bash
# 한 줄로 확인
python sanity_check.py && echo "제출 가능!" || echo "DESK REJECT"

# 결과 요약 보기
python sanity_check.py && cat sanity_check_results.json | jq '.summary'
```

## 상세 문서

📚 `SANITY_CHECK_README.md` 참조
