#this script is used to write the openwebtext

from datasets import load_dataset
import tiktoken
from pathlib import Path
import os
import torch
import random

dataset = load_dataset("openwebtext")
train_data=dataset['train']

enc=tiktoken.get_encoding('gpt2')
tok_limit=10000000 #think about it 10 million tokens means ten mil line array, which could be i/o intensive?
output_dir=Path("./GPT2-exploration/data/openwebtext")
output_dir.mkdir(exist_ok=True)

buffer = []
file_idx = 0
token_count = 0

for example in train_data:
    text = example["text"]
    tokens = enc.encode(text)
    len_token = len(tokens)

    if token_count + len_token >= tok_limit:
        output_path = output_dir / f"tokens_part_{file_idx}.pt"
        torch.save(torch.tensor(buffer, dtype=torch.long), output_path)

        file_idx += 1
        buffer = []
        token_count = 0

    buffer.extend(tokens)
    token_count += len_token

if buffer:
    output_path = output_dir / f"tokens_part_{file_idx}.pt"
    torch.save(torch.tensor(buffer, dtype=torch.long), output_path)
    
