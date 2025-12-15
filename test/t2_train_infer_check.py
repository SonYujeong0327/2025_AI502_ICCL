import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer

import json
from datetime import datetime

from models.dtmodel import DTConfig, DTForCausalLM
from models import DefaultTransformerConfig, DefaultTransformerForCausalLM

print(" STEP 1: Initializing Tokenizer and Model ".center(60, "="))
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__} | Vocab size: {len(tokenizer)}")

print("✅ Creating a smaller DTConfig for sanity check...")
base_model_config = DTConfig(
    vocab_size=len(tokenizer),
    hidden_size=128,                 
    intermediate_size=512,             
    num_hidden_layers=2,              
    num_attention_heads=4,          
    num_key_value_heads=2,            
    max_position_embeddings=256,     

    rms_norm_eps=1e-5,
    attention_bias=False,
    lambda_std_dev=0.1,
    attention_implementation="eager", 
)
print("✅ Custom model config created and updated with tokenizer's vocab size.")
print(json.dumps(base_model_config.to_dict(), indent=2))

model = DTForCausalLM(base_model_config)
model.resize_token_embeddings(len(tokenizer))
print(f"✅ Custom model initialized: {model.__class__.__name__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.to(torch.bfloat16)
print(f"✅ Model moved to device: {device}\n")


print(" STEP 2: Preparing Data and Hyperparameters ".center(60, "="))
learning_rate = 1e-3
epochs = 30
batch_size = 8
seq_length = 16

dummy_text = "This is a sanity check. " * 50
tokenized_text = tokenizer.encode(dummy_text)

inputs = [tokenized_text[i:i+seq_length] for i in range(len(tokenized_text) - seq_length)]
input_ids = torch.tensor(inputs, dtype=torch.long)
labels = input_ids.clone()
dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"✅ Dummy data created with {len(dataset)} samples.\n")

print(" STEP 3: Starting Training ".center(60, "="))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model.train()
training_losses = []

for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        b_input_ids, b_labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=b_input_ids, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    training_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

final_loss = training_losses[-1]
print(f"✅ Training complete. Final average loss: {final_loss:.4f}\n")

print(" STEP 4: Running Inference ".center(60, "="))
model.eval()
prompt = "This is a"
prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

with torch.no_grad():
    output_sequences = model.generate(
        input_ids=prompt_ids,
        max_length=100,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(f"Prompt: '{prompt}'")
print(f"Generated Text: '{generated_text}'\n")

print(" STEP 5: Saving Results to JSON ".center(60, "="))
results = {
    "timestamp": datetime.now().isoformat(),
    "model_config": base_model_config.to_dict(),
    "training_params": {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "device": str(device)
    },
    "training_results": {
        "loss_per_epoch": training_losses,
        "final_avg_loss": final_loss
    },
    "inference_results": {
        "prompt": prompt,
        "generated_text": generated_text
    }
}

output_filename = "train_infer_check_results.json"
with open(output_filename, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ All results have been saved to '{output_filename}'")
print(json.dumps(results, indent=2))