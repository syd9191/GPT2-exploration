from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F

#---------------------------------------------------------
@dataclass
class GPTConfig:
    block_size:int=1024
    vocab_size:int=50257
    n_layer:int=12
    n_head:int=12
    n_embd:int=768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head==0 
        self.c_attn=nn.Linear(config.n_embd, 3*config.n_embd) #for the k,q,v

        self.c_proj=nn.Linear(config.n_embd, config.n_embd) 
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head=config.n_head
        self.n_embd=config.n_embd

    def forward(self, x):
        B, T, C= x.size() #batch size, sequence length, embedding dimentionality
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv=self.c_attn(x) # x.shape = [batch, seq_len, n_embd] = [B, T, 768] - > [batch, seq_len, 3*n_embd] -> [B, T, 768] 

        """
        i was quite confused on how it was split, but apparently for each qkv token, the q, k, and v values are just concatenated, so when we call qkv.split(n_embd) on the last feature, we get the q, k and v 
        qkv[0, 0] = [q₀ q₁ q₂ q₃ q₄ q₅ q₆ q₇  |  k₀ k₁ k₂ k₃ k₄ k₅ k₆ k₇  |  v₀ v₁ v₂ v₃ v₄ v₅ v₆ v₇]
              <------ Q ------->       <------ K ------->       <------ V ------->
        """
        q, k, v=qkv.split(self.n_embd, dim=2) #[B, T, 768] again

        #here we split each KQV tensor into number of heads for simulatneous processing, we force a new tensor shape while keeping the same data so we can calc attention down the road
    
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #final output dimension [B,n_head,T,hs]
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #final output dimension [B,n_head,T,hs]
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #final output dimension [B,n_head,T,hs]

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

# ----------------
prompt="Hello, I'm a language model,"
num_repeat_sequences=5
max_length=30

#autodetection for device
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device="mps" 

print("Using device:", device)

#model=GPT.from_pretrained('gpt2')
model=GPT(GPTConfig())
print("Model Weights Load Done")

model.eval() #turns it into eval mode, which is good practice apparently
model.to(device)

import tiktoken
enc=tiktoken.get_encoding('gpt2')
tokens=enc.encode(prompt)
tokens=torch.tensor(tokens, dtype=torch.long)
tokens=tokens.unsqueeze(0).repeat(num_repeat_sequences, 1) #[5,8] #unsqueeze here adds a new dimension at pos 0, repeat replicates the same sequece 5 times
x=tokens.to(device)
torch.manual_seed(42)

"""
This whole part is still kind of confusing to me
"""

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

