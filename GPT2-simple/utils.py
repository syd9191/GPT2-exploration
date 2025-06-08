import torch 
from typing import Tuple

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