import random
import torch 
import os 
from pathlib import Path
from typing import Union
from torch.utils.data import IterableDataset

class DirStreamingDataset(IterableDataset):
    """
    openwebtext directory now contains pt files in binary, precomputed with the GPT 2 encodings

    Each file is 10 million tokens, total of 904 files
    """

    def __init__(self, 
                 path: Union[str, os.PathLike], 
                 block_size: int,
                 seed:int):
        self.dir_path = Path(path)
        if not self.dir_path.exists():
            raise FileNotFoundError(path)
        
        self.token_files=sorted(self.dir_path.glob("*.pt"))
        if not self.token_files:
            raise ValueError(f"No .pt files found in {self.dir_path}")
        self.num_files=len(self.token_files)
        self.block_size = block_size
        self.num_token_per_file=10000000
        self.seed=seed

        # introduce per-worker offset to avoid every worker seeing the
        # same sequence order when shuffling is disallowed.
        # DataLoader(worker_init_fn) will set this value.
        self._worker_offset = 0


    # DataLoader with an IterableDataset must NOT shuffle. We instead start each worker at a random offset to approximate randomness.
    def set_worker_offset(self, offset: int):
        self._worker_offset = offset

    def get_dataset_token_count(self):
        return self.num_files*self.num_token_per_file

    def __iter__(self):
        rng = torch.Generator().manual_seed(42 + self._worker_offset)
        file_indices = list(range(len(self.token_files)))
        random.shuffle(file_indices)

        while True:
            for idx in file_indices:
                tokens = torch.load(self.token_files[idx])
                n = len(tokens)
                pos = torch.randint(0, n - self.block_size - 1, (1,), generator=rng).item()

                while pos + self.block_size + 1 < n:
                    #now instead of randomising the start position, we randomise which file we see, then the start position is sequential
                    chunk = tokens[pos : pos + self.block_size + 1]
                    x = chunk[:-1]
                    y = chunk[1:]
                    pos += self.block_size
                    yield x, y# DataLoader → (B,T)

    def __len__(self):
        # Just return an estimate or token count divided by block size
        return self.get_dataset_token_count() // self.block_size     
