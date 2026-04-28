# utils.py
import gc
import torch

def clean_memory():
    """清理GPU内存"""
    torch.cuda.empty_cache()
    gc.collect()