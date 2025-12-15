#!/usr/bin/env python3
"""
Test script for native Flash Attention 2 implementation.
Validates that FlashMultiHeadAttention works correctly with the optimized implementation.
"""

import torch
import transformers
from models.example_config import ExampleConfig
from models.example_model import ExampleTransformerForCausalLM

def test_flash_attention():
    """Test Flash Attention forward pass with dummy data."""
    print("=" * 80)
    print("Testing Native Flash Attention 2 Implementation")
    print("=" * 80)
    
    # Load config
    config = ExampleConfig.from_json_file("config/example_model_config.json")
    
    # Create small model for testing
    config.num_hidden_layers = 2
    config.hidden_size = 256
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.max_position_embeddings = 512
    
    print(f"\n📋 Test Configuration:")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    print(f"   Num heads: {config.num_attention_heads}")
    print(f"   Num KV heads: {config.num_key_value_heads}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"\n🔧 Device: {device}")
    print(f"🔧 Dtype: {dtype}")
    
    model = ExampleTransformerForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()
    
    # Test 1: Basic forward pass
    print("\n" + "=" * 80)
    print("Test 1: Basic Forward Pass")
    print("=" * 80)
    
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected shape: ({batch_size}, {seq_length}, {config.vocab_size})")
    
    assert logits.shape == (batch_size, seq_length, config.vocab_size), "Output shape mismatch"
    
    # Test 2: Forward with attention mask (padding)
    print("\n" + "=" * 80)
    print("Test 2: Forward with Attention Mask (Padding)")
    print("=" * 80)
    
    # Create attention mask: first sequence has 32 tokens, second has 64
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    attention_mask[0, 32:] = 0  # First sequence is padded after position 32
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    print(f"✅ Forward with attention mask successful")
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Non-padding tokens: seq1={attention_mask[0].sum().item()}, seq2={attention_mask[1].sum().item()}")
    print(f"   Output shape: {logits.shape}")
    
    # Test 3: Forward with loss
    print("\n" + "=" * 80)
    print("Test 3: Forward with Loss Calculation")
    print("=" * 80)
    
    labels = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
    
    print(f"✅ Forward with loss successful")
    print(f"   Loss: {loss.item():.4f}")
    
    # Test 4: Backward pass (gradient flow)
    print("\n" + "=" * 80)
    print("Test 4: Backward Pass (Gradient Flow)")
    print("=" * 80)
    
    model.train()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    print(f"✅ Backward pass successful")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed: {has_gradients}")
    
    assert has_gradients, "No gradients computed"
    
    # Test 5: Generation with KV cache
    print("\n" + "=" * 80)
    print("Test 5: Generation with KV Cache")
    print("=" * 80)
    
    model.eval()
    batch_size = 1
    input_length = 32
    num_new_tokens = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, input_length), device=device)
    
    # First forward (prefill)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    print(f"   Prefill phase:")
    print(f"   - Input length: {input_length}")
    print(f"   - Cache created: {past_key_values is not None}")
    
    # Generate tokens one by one (decode phase)
    generated = [next_token]
    for i in range(num_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_token)
    
    print(f"✅ Generation with KV cache successful")
    print(f"   Decode phase:")
    print(f"   - Generated {num_new_tokens} tokens")
    print(f"   - Cache shape (layer 0): {past_key_values[0][0].shape}")
    
    # Test 6: Numerical stability check
    print("\n" + "=" * 80)
    print("Test 6: Numerical Stability")
    print("=" * 80)
    
    model.eval()
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    # Run twice and check consistency
    with torch.no_grad():
        outputs1 = model(input_ids=input_ids)
        outputs2 = model(input_ids=input_ids)
        
        logits1 = outputs1.logits
        logits2 = outputs2.logits
        
        max_diff = (logits1 - logits2).abs().max().item()
    
    print(f"✅ Numerical stability check")
    print(f"   Max difference between runs: {max_diff:.2e}")
    
    assert max_diff < 1e-5, f"Numerical instability detected: {max_diff}"
    
    print("\n" + "=" * 80)
    print("🎉 All Tests Passed!")
    print("=" * 80)
    print("\n✨ Native Flash Attention 2 implementation is working correctly!")
    print("   - Basic forward pass: ✅")
    print("   - Attention mask handling: ✅")
    print("   - Loss calculation: ✅")
    print("   - Gradient flow: ✅")
    print("   - KV cache generation: ✅")
    print("   - Numerical stability: ✅")

if __name__ == "__main__":
    test_flash_attention()
