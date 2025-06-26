import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
import tiktoken


#custom
import utils
from dataloaders.dir_streaming_dataset import DirStreamingDataset
from model.GPT2_builder import GPT
from config import GPTConfig
from model.model_tracker import ModelTracker

# ---------------- sampling loop, we could probably abstract this later on 
if __name__=="__main__":
    data_path     = "./GPT2-exploration/data/smol_smoltalk"
    model_name    = "gpt-2"
    B, T          = 16, 1024 #micro batch size
    max_lr        = 6e-4
    max_steps     = 100
    save_interval = 25
    batch_size    = 524288 #2**19 we stick with the pow 2 again
    enc           = tiktoken.get_encoding("gpt2")
    prompt        = "Describe the colour yellow?</s>"
    mode          = "eval"

    assert batch_size%(B*T)==0 

    grad_accum_steps=batch_size//(B*T)

    plot_path   = f"./GPT2-exploration/analytics/{model_name}/plots/{model_name}_finetuning_loss_curve.png"
    model_path  = f"./GPT2-exploration/models/{model_name}/{model_name}.pth"
    stats_path  = f"./GPT2-exploration/analytics/{model_name}/stats/{model_name}_training_stats.json"
    loss_path   = f"./GPT2-exploration/analytics/{model_name}/stats/{model_name}_finetuning.npy"

    device, use_amp_cuda = utils.get_device(verbose=True)

    if mode=="train":
        dir_path=Path(data_path)

        

        dataset = DirStreamingDataset(path=data_path, 
                                    block_size=T,
                                    seed=42)
        val_dataset = DirStreamingDataset(path=data_path, 
                                        block_size=T,
                                        seed=24) #another instance for val use

        torch.set_float32_matmul_precision('high')

        n_tokens= dataset.get_dataset_token_count()
        epoch_size= int(n_tokens/(B*T))
        print(f"epoch_size: {epoch_size}")

        if torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            sampler = DistributedSampler(dataset, shuffle=False)
            val_sampler= DistributedSampler(val_dataset, shuffle=False)
            ddp=True
        else:
            sampler = None
            ddp=False

        loader = utils.make_loader(
            dataset,
            batch_size=B,
            num_workers=4,            # no separate workers ➜ no pickling
            sampler=sampler,
            device_is_cuda=device.type == "cuda",
        )

        val_loader = utils.make_loader(
            val_dataset,
            batch_size=B,
            num_workers=4,            # no separate workers ➜ no pickling
            sampler=val_sampler,
            device_is_cuda=device.type == "cuda",
        )

        utils.smoke_test(B=B,
                        T=T,
                        loader=loader) #test before training

        model = GPT(config=GPTConfig(),
                    sampler=sampler,
                    enc=enc).to(device)
        model = torch.compile(model)

        optimiser = model.configure_optimizers(weight_decay=0.1,
                                            learning_rate=max_lr
                                            )
        

        try:
            print(f"Loading weights from: {model_path}")
            model.load_weights(path=model_path,
                            device=device,
                            weights_only=True)
        except Exception as e:
            print(f"No pre-trained weights found at {model_path}. Starting from scratch.")
            print(f"Error: {e}")
        

        if torch.cuda.device_count() > 1:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device()
            ) # sync gradients

        scaler_device   = "cuda" if device.type == "cuda" else "cpu"
        use_device_amp  = device.type == "cuda"               # enable only on CUDA
        scaler          = torch.amp.GradScaler(scaler_device,
                                        enabled=use_device_amp)

        model.train()

        model_tracker=ModelTracker(model=model,
                                batch_size=B,
                                seq_length=T,
                                total_tokens=n_tokens,
                                plot_path=plot_path,
                                model_path=model_path,
                                stats_path=stats_path,
                                loss_path=loss_path )
        

        model.module.training_loop(max_steps=max_steps,
                    save_interval=save_interval,
                    batch_size=B, 
                    seq_len=T, 
                    max_lr=max_lr, 
                    test_prompt=prompt,
                    grad_accum_steps=grad_accum_steps,
                    use_amp_cuda=use_amp_cuda,
                    ddp=ddp, 
                    loader=loader, 
                    val_loader=val_loader, 
                    model_tracker=model_tracker,
                    device=device.type,
                    scaler=scaler,
                    optimiser=optimiser)

        model.module.invoke(prompt=prompt,
                    max_new_tokens=30, 
                    device=device, 
                    )

        import  sys;sys.exit(0)

    elif mode=="eval":
        model = GPT(config=GPTConfig(),
                    sampler=None,
                    enc=enc).to(device)
        model = torch.compile(model)

        try:
            print(f"Loading weights from: {model_path}")
            model.load_weights(path=model_path,
                            device=device,
                            weights_only=True)
        except Exception as e:
            print(f"No pre-trained weights found at {model_path}. Starting from scratch.")
            print(f"Error: {e}")

        response=model.invoke(prompt=prompt,
                    max_new_tokens=300, 
                    device=device, 
                    )
        
        print(response)
        

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

