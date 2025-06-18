from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size:int=2048
    vocab_size:int=50304
    n_layer:int=12
    n_head:int=12
    n_embd:int=768