import os, random
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import tiktoken

#---------------------------------------------------------
@dataclass
class GPTConfig:
    block_size:int=1024
    vocab_size:int=50257
    n_layer:int=12
    n_head:int=12
    n_embd:int=768

class TokenTextDataset(IterableDataset):
    """
    Streams the entire file token-by-token forever, returning sa contiguous
    sliding window of `block_size` for x and the next-token labels for y.
    """

    def __init__(self, path: str | os.PathLike, block_size: int):
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
            yield x, y                                                      # DataLoader → (B,T)

class DataLoaderLite():
    def __init__(self, B, T, device):
        self.device=device
        self.B=B
        self.T=T
        with open('../exploration/input.txt', 'r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} Tokens")
        print(f"1 Epoch = {len(self.tokens)//(B*T)} Batches")
        self.current_pos=0

    def get_next_batch(self):
        B=self.B
        T=self.T
        buf=self.tokens[self.current_pos: self.current_pos+(B*T)+1] #similar batching that we tested
        x=buf[:-1].view(B, T).to(self.device) #should follow a B batches of T sequences kind of format
        y=buf[1:].view(B, T).to(self.device)
        self.current_pos+=B*T
        if self.current_pos + (B*T+1)>=len(self.tokens): #we have already reached the end of our training dataset
            self.current_pos=0
        return x, y
        


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head==0 
        self.c_attn=nn.Linear(config.n_embd, 3*config.n_embd) #for the k,q,v

        self.c_proj=nn.Linear(config.n_embd, config.n_embd) 
        self.c_proj.NANOGPT_SCALE_INIT = 1 #this is a flag for scaling down weights, following xaviers initialisation
        self.n_head=config.n_head
        self.n_embd=config.n_embd

    def forward(self, x):
        B, T, C= x.size() #batch size, sequence length, embedding dimentionality
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv=self.c_attn(x) # x.shape = [batch, seq_len, n_embd] = [B, T, 768] - > [batch, seq_len, 3*n_embd] -> [B, T, 3*768] 

        """
        i was quite confused on how it was split, but apparently for each qkv token, the q, k, and v values are just concatenated, so when we call qkv.split(n_embd) on the last feature, we get the q, k and v 
        qkv[0, 0] = [q₀ q₁ q₂ q₃ q₄ q₅ q₆ q₇  |  k₀ k₁ k₂ k₃ k₄ k₅ k₆ k₇  |  v₀ v₁ v₂ v₃ v₄ v₅ v₆ v₇]
              <------ Q ------->       <------ K ------->       <------ V ------->
        """
        q, k, v=qkv.split(self.n_embd, dim=2) #[B, T, 768] again

        #here we split each KQV tensor into number of heads for simulatneous processing, we force a new tensor shape while keeping the same data so we can calc attention down the road
    
        k=k.contiguous().view(B,T,self.n_head,C//self.n_head).transpose(1,2) #final output dimension [B,n_head,T,hs]
        q=q.contiguous().view(B,T,self.n_head,C//self.n_head).transpose(1,2) #final output dimension [B,n_head,T,hs]
        v=v.contiguous().view(B,T,self.n_head,C//self.n_head).transpose(1,2) #final output dimension [B,n_head,T,hs]

        y=F.scaled_dot_product_attention(q,k,v, is_causal=True) #normal dot prod calculation of attention, this one might use flash_attn not very sure
        y=y.transpose(1,2).contiguous().view(B,T,C) #reassembles the heads back to the original dimensions
        
        y=self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh') #GELU: Gausian Error Linear Unit, is what they came up with to deal with the dead neuron problem that comes with RELU, read more here: http://arxiv.org/pdf/1606.08415v5
        self.c_proj=nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x=self.c_fc(x) ##pass through the fully connect
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self, x): #feedforward process similar to attention is all you need
        x= x + self.attn(self.ln_1(x)) #add residual connection, originating from RESNET 50 actually
        x= x + self.mlp(self.ln_2(x))  #add residual connection
        return x

class GPT(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #h is the number of heads
            ln_f=nn.LayerNorm(config.n_embd), #layer norm
        ))
        self.lm_head=nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing: This concept is expanded on the google docs: https://docs.google.com/document/d/1cRYtDPcxogKBLilWpaeLZIxQlXp04IAZwKnCuk3GhUU/edit?tab=t.0
        self.transformer.wte.weight=self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std*= (2*self.config.n_layer)**-0.5 #2 cause there are two residual connections for each block i guess
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    
    def forward(self, idx, targets=None):
        B, T=idx.size() #Batch , Sequence length
        assert T<=self.config.block_size, f"Cannot forward sequence of length: {T}, with a block size of only {self.config.block_size}"
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device) #this is just a tensor shape [T]
        pos_emb=self.transformer.wpe(pos)
        tok_embd=self.transformer.wte(idx) #this should produce something like [B, T, n_embd], for each token in the sequence, it should make it into a 768 dimension vector embedding
        x=pos_emb+tok_embd
        
        for block in self.transformer.h:
            x=block(x) #forward across all transformer blocks

        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) #(B,T,vocab_size) for each token, this is the confidence score for each word in the vocab being the next token, pre softmax
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits,loss

    @classmethod
    def from_pretrained(cls, model_type):
        """ HONESTLY NOT GOING TO SPEND THE TIME TO UNDERSTAND PARAM LOADING
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class ModelTracker:
    def __init__(self, 
                model:torch.nn.Module,
                plot_path:str, 
                model_path:str):

        self.losses:list=[]
        self.best_loss:float=float('inf')
        self.plot_path:str=plot_path
        self.model_path:str=model_path 
        self.best_weights=None
        self.model=model

    def add_loss(self, loss):
        self.losses.append(loss)
        if loss<self.best_loss and self.model is not None:
            self.best_loss=loss
            self.best_weights=self.model.state_dict()

    def save_best_weights(self):
        if self.best_weights is not None:
            torch.save(self.best_weights, self.model_path)
    
    def plot_loss(self, show=True):
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300)
        if show:
            plt.show()
        

def make_loader(dataset: TokenTextDataset, batch_size: int, num_workers: int,
                sampler: DistributedSampler | None, device_is_cuda: bool):
    def worker_init_fn(worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        dataset: TokenTextDataset = worker_info.dataset
        offset = random.randint(0, len(dataset.tokens) - 1)
        dataset.set_worker_offset(offset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device_is_cuda,
        prefetch_factor=(2 if num_workers > 0 else None), # does not run when 0 num_workers
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

# ---------------- sampling loop, we could probably abstract this later on 
if __name__=="__main__":
    data_path   = "./exploration/input.txt"
    plot_path   = "./plots/training_loss_curve.png"
    model_path  = "./models/gpt-2-tinyshakespeare.pth"
    B, T        = 4, 32
    lr          = 3e-4
    max_steps   = 1000


    if torch.cuda.is_available():
        device        = torch.device("cuda")
        use_amp_cuda  = True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device        = torch.device("mps")
        use_amp_cuda  = False
    else:
        device        = torch.device("cpu")
        use_amp_cuda  = False

    print("Using device:", device)

    dataset = TokenTextDataset(path=data_path, block_size=T)

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    loader = make_loader(
        dataset,
        batch_size=B,
        num_workers=0,            # no separate workers ➜ no pickling
        sampler=sampler,
        device_is_cuda=device.type == "cuda",
    )

    model = GPT(GPTConfig()).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device()
        ) # sync gradients

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler_device   = "cuda" if device.type == "cuda" else "cpu"
    use_device_amp  = device.type == "cuda"               # enable only on CUDA
    scaler          = torch.amp.GradScaler(scaler_device,
                                     enabled=use_device_amp)

    model.train()

    model_tracker=ModelTracker(model=model,
                            plot_path=plot_path,
                            model_path=model_path)
    
    for step, (x, y) in enumerate(loader):
        if step >= max_steps:
            break

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, enabled=use_amp_cuda):
            logits, loss = model(x, y)

        model_tracker.add_loss(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        if sampler is not None:
            sampler.set_epoch(step)  # shuffle shards every epoch


        print(f"step {step:6d}  loss {loss.item():.4f}")

    print("training loop finished ✔")

    model_tracker.plot_loss(show=True)
    model_tracker.save_best_weights()

    model.eval()
    prompt = "Hello, I'm a language model,"
    enc = tiktoken.get_encoding("gpt2")
    x   = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, :]
    torch.manual_seed(42)
    while x.size(1) < 50:
        with torch.no_grad():
            logits, _ = model(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
    print(enc.decode(x[0].tolist()))

    def smoke_test(loader, n_batches: int = 3):
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

    smoke_test(loader)
    import  sys;sys.exit(0)

    """
    #use this code if we wanna hack generation loop
    import tiktoken
    enc=tiktoken.get_encoding('gpt2')
    tokens=enc.encode(prompt)
    tokens=torch.tensor(tokens, dtype=torch.long)
    tokens=tokens.unsqueeze(0).repeat(num_repeat_sequences, 1) #[5,8] #unsqueeze here adds a new dimension at pos 0, repeat replicates the same sequece 5 times
    x=tokens.to(device)
    torch.manual_seed(42)
   

    while x.size(1)<max_length: #this loop starts when inputting a sequence of tokens, with whatever is input, whole ass transformer will look back and predict next token
        with torch.no_grad():
            #take note that x is in shape [B,T] right now
            logits, loss=model(x)  #after running through the model [B,T,vocab_size], for each token, we have the logits for the entire vocab, and this logit represents the confidence for this token to be the next token
            logits=logits[:, -1, :] #we zoom in on the logits for the last token, now logits shape is [B, vocab_size]
            probs=F.softmax(logits, dim=-1) 
            topk_probs, topk_indices=torch.topk(probs, k=50, dim=-1) #shape [B, k] which is the number of batches and the top k tokens, and their probabilities
            ix=torch.multinomial(topk_probs, 1) #next selected index will be random based on the top 50 probabilities
            xcol=torch.gather(topk_indices, 1, ix) #match the token that is selected
            x=torch.cat((x,xcol), dim=1) #concats the x tokens with the newest one


    for i in range(num_repeat_sequences): #this part just prints all of the different repeat sequences, if k=1, all sequences will be the same
        tokens=x[i, :max_length].tolist()
        decoded=enc.decode(tokens)
        print(">", decoded)

    """

