import os
import torch
import torch.nn as nn
from model.GPT2_builder import GPT, GPTConfig
from hellaswag import load_model, evaluate

'''
JUST PUT EVERYTHING IN THE HELLASWAG
'''

def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Dynamic quantization of all linear layers only, leaves all other layers like LayerNorm and Embeddings in f32
    """
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

def main():
    device = torch.device("cpu")
    # only purpose of model_path is to get the size, the path seems to be hardcoded in the load_model in hellaswag.py anyway and not an attribute anywhere
    model_path = "../models/gpt-2/gpt-2.pth"
    model = load_model("self_train", device)
    model.eval()

    q_model = quantize_model(model)
    quant_path = "../models/gpt-2/gpt2-quantized.pth"
    torch.save(q_model.state_dict(), quant_path)
    print(f"Saved quantized model to {quant_path}")

    size_fp32 = os.path.getsize(model_path) / 1e6
    size_q = os.path.getsize(quant_path) / 1e6
    print(f"FP32 model: {size_fp32:.2f} MB")
    print(f"Quantized model: {size_q:.2f} MB → {(size_q/size_fp32 * 100):.1f}% of original\n")

    # acc_fp32 = evaluate(model, device) tentatively not running both since my laptop is a heaping pile of trash, hard coded number from Stinknee
    acc_fp32 = 28.6
    acc_q = evaluate(q_model, device)
    print(f"HellaSwag accuracy: FP32 {acc_fp32:.4f}, Quantized {acc_q:.4f}")

if __name__=="__main__":
    main()
