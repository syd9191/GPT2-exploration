import tiktoken
import torch 
import os 
from pathlib import Path
from typing import Union
from torch.utils.data import IterableDataset


class TokenTextDataset(IterableDataset):
    """
    NOT IN USE
    Streams the entire file token-by-token forever, returning sa contiguous
    sliding window of `block_size` for x and the next-token labels for y.
    """

    def __init__(self, path: Union[str,os.PathLike], block_size: int):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        text      = path.read_text(encoding="utf-8")
        enc       = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        self.block_size = block_size

        # introduce per-worker offset to avoid every worker seeing the
        # same sequence order when shuffling is disallowed.
        # DataLoader(worker_init_fn) will set this value.
        self._worker_offset = 0

    # DataLoader with an IterableDataset must NOT shuffle. We instead start each worker at a random offset to approximate randomness.
    def set_worker_offset(self, offset: int):
        self._worker_offset = offset

    def get_dataset_token_count(self):
        return len(self.tokens)

    def __iter__(self):
        n   = len(self.tokens)
        pos = self._worker_offset % n
        rng = torch.Generator().manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

        while True:
            # wrap & optionally jump to a new random start to decorrelate batches
            if pos + self.block_size + 1 >= n:
                pos = torch.randint(0, n - self.block_size - 2, (1,), generator=rng).item()

            chunk  = self.tokens[pos : pos + self.block_size + 1]           # (T+1,)
            x      = chunk[:-1]                                             # (T,)
            y      = chunk[ 1:]                                             # (T,)
            pos   += self.block_size
            yield x, y    
            
                                             # DataLoader → (B,T)
