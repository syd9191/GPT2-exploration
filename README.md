# GPT2-Exploration

This project implements a custom GPT-2-style language model pipeline from scratch using PyTorch. It features full training, evaluation, logging, distributed training support, and token-level streaming from disk. The setup is modular and research-friendly, designed for experimentation with scaling and architecture changes.

---

## ✅ Features Overview

### 🧠 Model
- GPT-2-style Transformer decoder blocks
- Token and position embeddings (`wte`, `wpe`)
- Configurable architecture via `GPTConfig`
- `torch.compile` integration for optimization

### 🛠️ Training
- Cosine LR decay with linear warmup
- Mixed-precision training (`torch.amp`)
- Optional gradient accumulation
- DDP-compatible for multi-GPU training
- Gradient clipping and norm tracking

### 📦 Data Pipeline
- Stream large tokenized datasets from disk
- Custom `DirStreamingDataset` (no memory blowup)
- Deterministic train/val split via seed

### 📉 Logging & Evaluation
- Sample generations printed at regular intervals
- Periodic val loss computation
- Track TPS, gradient norms, LR, and loss
- Outputs: `.pth` model, `.json` stats, `.npy` loss array, `.png` loss curve

---

## 🛠️ Setup

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training
```bash
python gpt2.py --mode train
```

### Evaluation / Inference
```bash
python gpt2.py --mode eval
```

---

## 🔢 Configurations

- `B`, `T` = Microbatch size and sequence length
- `batch_size` = Total tokens per step (via grad accumulation)
- `max_steps` = Total training steps
- `max_lr` = Peak learning rate

These are auto-handled:
```python
grad_accum_steps = batch_size // (B * T)
```

---

## 🏗️ Structure

```
GPT2-exploration/
├── Data/
│   ├── openwebtext/ 
│   │   ├── tokens_part_x.pt
│   ├── smol-smoltalk/
│   │   ├── tokens_part_x.pt
│   ├── openwebtext.py #run this to create openwebtext token DIR
│   ├── openwebtext.py #run this to create smol-smoltalk token DIR
├── GPT2-simple/
│   ├── gpt2.py
│   ├── model/
│   │   ├── GPT2_builder.py
│   │   └── model_tracker.py
│   ├── dataloaders/
│   │   ├── dataloader_lite.py (DEPRECIATED)
│   │   ├── token_text_dataset.py (DEPRECIATED)
│   │   └── dir_streaming_dataset.py
│   ├── hellaswag/
│   │   └── hellaswag_val.jsonl
│   ├── config.py
│   ├── hellaswag.py       # Hellaswag Eval
│   └── utils.py
├── models/
│   └── gpt-2/
│   │   └── gpt-2.pth
├── analytics/
│   └── gpt-2/
│       ├── plots/         # Loss curves
│       └── stats/         # Logs and losses
├── requirements.txt
```

---

## 🧭 Roadmap

- [ ] Scale Model Architechture
- [ ] Add FlashAttention 2 support
- [ ] Switch to SwiGLU + RMSNorm
- [ ] Incorporate Rotary Positional Embeddings
- [ ] ONNX export and HF compatibility
- [ ] Multiple dataset mixing + shuffling
- [ ] More Analytics!!! Validation Accuracy plotting

---

## ⚠️ Notes
- For DDP, use `torchrun` or `accelerate` to launch across GPUs
- When using `DistributedDataParallel`, access model as `model.module`
- Ensure vocab size and sequence length match when resuming from checkpoint
- Supports graceful `KeyboardInterrupt` exits with checkpoint saving

---

