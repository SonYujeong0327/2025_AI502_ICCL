# Flash Attention 2 Implementation - Quick Reference

## 개요

`models/example_layers.py`의 `FlashAttentionMixin`과 `FlashMultiHeadAttention`은 HuggingFace Transformers의 래퍼를 사용하지 않고 **native Flash Attention 2 API를 직접 호출**합니다.

## 핵심 변경사항

### 1. Direct API 호출 (Before → After)

**Before (Old):**
```python
attn_output, _ = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
    self, query_states, key_states, value_states, attention_mask, ...
)
```

**After (Native):**
```python
from flash_attn import flash_attn_func, flash_attn_varlen_func

attn_output = FlashAttentionMixin.forward(
    query=query_states,
    key=key_states,
    value=value_states,
    attention_mask=attention_mask,
    ...
)
```

### 2. Transpose 최적화

**Before:**
```python
# 3번 transpose: projection → FA2 → output
query = self.q_proj(x).view(...).transpose(1, 2)  # transpose 1
attn_out = fa2_wrapper(query, ...)  # 내부에서 transpose 2
output = attn_out.transpose(1, 2).reshape(...)  # transpose 3 (내부)
```

**After:**
```python
# 2번 transpose (RoPE만): projection → RoPE → FA2 → output
query = self.q_proj(x).view(...)  # (B, S, H, D) - FA2 형식 유지
query_rope = query.transpose(1, 2)  # RoPE를 위한 임시 transpose
apply_rope(query_rope, ...)
query = query_rope.transpose(1, 2)  # 다시 FA2 형식
attn_out = flash_attn_func(query, ...)  # transpose 없이 직접 사용
```

### 3. Varlen 지원 (Padding 최적화)

```python
if attention_mask is not None:
    # 1. Unpad: padding 제거
    q, k, v, indices, cu_seqlens, max_seqlens = _upad_input(...)
    
    # 2. Varlen FA2: 실제 토큰만 계산
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
    
    # 3. Pad: 원래 형식으로 복원
    out = pad_input(out, indices, batch, seq_len)
else:
    # No padding: 표준 함수 사용
    out = flash_attn_func(query, key, value, ...)
```

## 클래스 구조

### FlashAttentionMixin

| Method | Description |
|--------|-------------|
| `_get_unpad_data()` | Attention mask → indices, cu_seqlens, max_seqlen |
| `_upad_input()` | Q, K, V에서 padding 제거 + metadata 생성 |
| `forward()` | Native FA2 API 호출 (varlen/standard 자동 선택) |

### FlashMultiHeadAttention

```python
class FlashMultiHeadAttention(MultiHeadAttention):
    def forward(self, hidden_states, position_embeddings, ...):
        # 1. Project Q, K, V (FA2 형식: B, S, H, D)
        q = self.q_proj(hidden_states).view(...)
        k = self.k_proj(hidden_states).view(...)
        v = self.v_proj(hidden_states).view(...)
        
        # 2. Apply RoPE (임시 transpose)
        q_rope = q.transpose(1, 2)
        k_rope = k.transpose(1, 2)
        apply_rope(q_rope, k_rope, cos, sin)
        q = q_rope.transpose(1, 2)
        k = k_rope.transpose(1, 2)
        
        # 3. Handle KV cache (cache 형식: B, H, S, D)
        if past_key_values is not None:
            k_cache = k.transpose(1, 2)
            v_cache = v.transpose(1, 2)
            k_cache, v_cache = past_key_values.update(...)
            k = k_cache.transpose(1, 2)
            v = v_cache.transpose(1, 2)
        
        # 4. Flash Attention (FA2 형식 유지)
        attn_out = FlashAttentionMixin.forward(q, k, v, ...)
        
        # 5. Output projection
        return self.o_proj(attn_out.reshape(...))
```

## 성능 결과

| Metric | Improvement |
|--------|-------------|
| Training Speed | **1.18x faster (15.4%)** |
| Training Memory | **18.2% less (29.9 MB)** |
| Inference Speed | **1.13x faster (11.3%)** |
| Inference Memory | **3.8 MB less** |
| Generation Latency | **1.09x faster (8.5%)** |

**Test config:** batch_size=2, seq_length=256, bf16, no-compile

## 요구사항

1. **Data type:** bf16 또는 fp16 (fp32 불가)
   ```python
   model = model.to(dtype=torch.bfloat16)
   ```

2. **Input shape:** `(batch, seq_len, num_heads, head_dim)`
   - PyTorch 표준 `(batch, num_heads, seq_len, head_dim)`과 다름

3. **Flash Attention 2.8.3+**
   ```bash
   pip install flash-attn>=2.8.3
   ```

## 테스트

```bash
# 기능 테스트
python test_flash_attention.py

# 성능 벤치마크
python benchmark.py --batch-size 2 --seq-length 256

# Default model 검증
python sanity_check.py
```

## 핵심 최적화 포인트

1. ✅ **HuggingFace wrapper bypass** → 3-5% 오버헤드 제거
2. ✅ **Transpose 감소** → 메모리 복사 최소화
3. ✅ **Varlen support** → Padding 토큰 계산 제거
4. ✅ **Fused operations** → Intermediate activation 감소

## 참고 파일

- **구현:** `models/example_layers.py` (lines 22-260)
- **상세 문서:** `FLASH_ATTENTION_IMPLEMENTATION.md`
- **테스트:** `test_flash_attention.py`
- **벤치마크:** `benchmark.py`
- **참조:** `reference/transformers/src/transformers/modeling_flash_attention_utils.py`

## Shape Reference

```
Input:  (B, S, D_model)
  ↓ projection
Q, K, V: (B, S, H, D_head)  ← Flash Attention 형식
  ↓ transpose for RoPE
Q, K (RoPE): (B, H, S, D_head)
  ↓ transpose back
Q, K: (B, S, H, D_head)
  ↓ Flash Attention (no transpose!)
Output: (B, S, H, D_head)
  ↓ reshape
Output: (B, S, D_model)
```

**Key:** Flash Attention 형식 `(B, S, H, D)`를 기본으로 유지하고, RoPE와 Cache 처리 시에만 임시 transpose.

---

**더 자세한 내용:** `FLASH_ATTENTION_IMPLEMENTATION.md` 참조
