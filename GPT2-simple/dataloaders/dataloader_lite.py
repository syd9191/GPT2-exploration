import torch
import tiktoken

class DataLoaderLite():
    """
    NOT IN USE
    """
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
        