import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tiktoken.core import Encoding
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist



# ------- Custom -------
from config import GPTConfig
from model.model_tracker import ModelTracker
import utils


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

        y=F.scaled_dot_product_attention(q,k,v, is_causal=True) #normal dot prod calculation of attention, Yes this one uses flash attention
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
    def __init__(self, 
                 config:GPTConfig,
                 enc:Encoding, 
                 sampler:DistributedSampler): 
        super().__init__()

        self.config=config
        self.sampler=sampler
        self.enc=enc


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
    
    def load_weights(self,
                    path:str,
                    device:str="cpu",
                    strict:bool=True, 
                    weights_only:bool=True):
        
        def remove_module_prefix(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module._orig_mod."):
                    new_key = k[len("module._orig_mod."):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            return new_state_dict
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file path: {path} does not exist")
        
        checkpoint = torch.load(path,map_location=device)  # or your device
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = remove_module_prefix(state_dict)
        self.load_state_dict(state_dict, strict=strict)
        print(f"Model weights loaded from {path}")

    def configure_optimizers(self, 
                             weight_decay, 
                             learning_rate):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
            print("Using fused AdamW.")
        except TypeError as e:
            print("Fused AdamW not supported, using unfused kernal.")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
    
    def val_loss(self,
                val_loader:DataLoader,
                device:str,
                ddp:bool
                ):
        """
        Gets Val Loss by evaluating 20 random sequences, no backprop on this function
        """
        val_loss_accum=0.0
        val_loss_steps=20
        self.eval()

        with torch.no_grad():
            for step, (x,y) in enumerate(val_loader):
                if step>=val_loss_steps:
                    break #we only need one example
                x,y=x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                    logits, loss=self(x,y)
                loss=loss/val_loss_steps
                val_loss_accum+=loss.detach()

        avg_loss = torch.tensor(val_loss_accum / val_loss_steps, device=device)

        if ddp:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

        if not ddp or (dist.is_initialized() and dist.get_rank()==0):
            print(f'Val Loss: {avg_loss.item():.4f}')

        self.train()
        


    def training_loop(self,
                      max_steps:int,
                      save_interval:int, 
                      batch_size:int,
                      seq_len:int,
                      test_prompt:str,
                      device:str, 
                      grad_accum_steps:int,  
                      use_amp_cuda:bool, 
                      ddp:bool, 
                      loader:DataLoader, 
                      val_loader:DataLoader, 
                      model_tracker:ModelTracker,
                      scaler:torch.amp.GradScaler,
                      optimiser:torch.optim
                      ):
        """
        I dont think training loop should be in the GPT2 class specifically, for convenience I put it here first 

        TODO: Model Training class
        """
        try:
            self.train()
            t0=time.time()
            step=0

            for batch_idx, (x, y) in enumerate(loader):
                if step >= max_steps:
                    break

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp_cuda):
                    logits, loss = self(x, y)
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    if step%100==0:
                        self.val_loss(val_loader=val_loader,
                                    device=device,
                                    ddp=ddp,
                                    )
                        model_response=self.invoke(max_new_tokens=30,
                                    prompt=test_prompt,
                                    device=device)
                        if not ddp or (dist.is_initialized() and dist.get_rank()==0):
                            print(model_response)
                        self.train()

                    norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)

                    # Logging and tracking
                    model_tracker.add_loss(loss.item() * grad_accum_steps)
                    lr = utils.get_lr(step=step, max_steps=max_steps)
                    for param_group in optimiser.param_groups:
                        param_group['lr'] = lr

                    torch.cuda.synchronize()
                    t1 = time.time()
                    tps = (batch_size * seq_len * grad_accum_steps) / (t1 - t0)
                    print(f"step: {step:6d} | loss: {loss.item() * grad_accum_steps:.4f} | tps: {tps:.2f} | dt: {t1-t0} | norm: {norm:.2f} | lr: {'{0:.2E}'.format(lr)}")

                    step += 1  # increment only after optimizer step
                    t0 = time.time()

                if step != 0 and step % save_interval == 0:
                        model_tracker.update_all(time_interval=(t1 - t0), curr_step=batch_idx)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user! Saving and plotting...")

        finally:
            t1=time.time()
            time_interval=t1-t0
            model_tracker.update_all(time_interval=time_interval,
                                    curr_step=batch_idx)
            
    def invoke(self, 
               max_new_tokens:int, 
               prompt:str, 
               device:str
               ):
        """
        Prompt the model, now only supports Next Token Sampling
        """
        self.eval()
        enc=self.enc

        x   = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, :]
        prompt_length=x.size(1)
        torch.manual_seed(42)

        while x.size(1) < prompt_length+max_new_tokens:
            #generation loop
            with torch.no_grad():
                logits, _ = self(x)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)

        return enc.decode(x[0].tolist())
