from datasets import load_dataset
import tiktoken
from pathlib import Path
import os
import torch
import random
import pprint
import tiktoken

ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")

import tiktoken

enc = tiktoken.get_encoding('gpt2')

buffer=[]
print(len(ds))
output_dir=Path("./data/smol-smoltalk")
file_idx=0
token_count=0
max_tokens = 10_240_000

for i in range(len(ds)):
    # Step 1: flatten all messages into one long token list per dataset item
    print(i)
    print(token_count)
    all_tokens = []
    for message in ds[i]['messages']:
        role = message['role']
        content = message['content']

        if role == 'user':
            text = "<s>" + content + "</s>"
        else:
            text = "<s>" + content + "</s>"

        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        len_token=len(tokens)

        if token_count+len_token>=max_tokens:
            output_path = output_dir / f"tokens_part_{file_idx}.pt"
            torch.save(torch.tensor(buffer, dtype=torch.long), output_path)

            file_idx += 1
            buffer = []
            token_count = 0

        buffer.extend(tokens)
        token_count+=len_token

if buffer:
    output_path = output_dir / f"tokens_part_{file_idx}.pt"
    torch.save(torch.tensor(buffer, dtype=torch.long), output_path)
    