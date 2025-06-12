import torch 
from typing import Tuple
import math 

def get_device(verbose:bool=True)->Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        device        = torch.device("cuda")
        use_amp_cuda  = True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device        = torch.device("mps")
        use_amp_cuda  = False
    else:
        device        = torch.device("cpu")
        use_amp_cuda  = False
    if verbose:
        print("Using Device: ", device)

    return device, use_amp_cuda


def smoke_test(loader, 
               B:int,
               T:int,
               n_batches: int = 3,
               ):
        print("running loader smoke-test:")
        for step, (x, y) in enumerate(loader):
            print(f"\nBatch {step}  ------------------------------------")
            print(f"x shape {tuple(x.shape)}, dtype {x.dtype}")
            print(f"y shape {tuple(y.shape)}, dtype {y.dtype}")

            # shape & dtype checks
            assert x.shape == (B, T)
            assert y.shape == (B, T)
            assert x.dtype == torch.long and y.dtype == torch.long

            # target-shift check:  y[b, t]  must equal  x[b, t+1]
            same = torch.all(y[:, :-1] == x[:, 1:])
            if not same:
                idx = (y[:, :-1] != x[:, 1:]).nonzero(as_tuple=False)[0]
                b, t = idx.tolist()
                raise ValueError(f"y is not a left-shifted copy of x at (batch={b}, pos={t})")
            print("✓ shapes & left-shift OK")

            # simple uniqueness check – tokens should not all be identical
            uniq = torch.unique(x).numel()
            print(f"unique tokens in x: {uniq}")
            assert uniq > 1, "looks like the same token repeated – did shuffling break?"

            if step + 1 == n_batches:
                break

        print("\n**** DataLoader smoke-test passed ****")


def get_lr( step:int, 
            max_steps:int, 
            warmup_steps:int=10,
            max_lr:float=3e-4):
        """
        Follows GPT-3's learning rate with a cosine decay

        From the paper:
        we clip the global norm of the gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260
        billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375
        million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over
        the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during
        training (until an epoch boundary is reached) to minimize overfitting. 
        """
        min_lr=max_lr*0.1

        if step<warmup_steps:
            return max_lr*(step+1)/warmup_steps
        elif step>max_steps:
            return min_lr
        decay_ratio=(step-warmup_steps)/(max_steps-warmup_steps)

        assert 0<=decay_ratio<=1
        coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))

        return min_lr+coeff*(max_lr-min_lr)