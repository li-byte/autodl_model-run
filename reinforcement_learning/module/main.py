# main.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
from datasets import load_dataset, Dataset
from swanlab.integration.transformers import SwanLabCallback

from config import LORA_RANK, SEED, MAX_SEQ_LENGTH, SYSTEM_PROMPT
from model_utils import init_model
from data_utils import format_sft_dataset
from train_sft import train_sft
from train_grpo import train_grpo
from utils import clean_memory

# -----------------------------
# 主函数
# -----------------------------
def main():
    # 初始化模型
    model, tokenizer = init_model("../grpo_saved_lora")

    # -------------------------
    # 配置 SwanLab 回调
    # -------------------------
    os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj"
    swanlab_callback = SwanLabCallback(
        project="Qwen3-VL-finetune",
        experiment_name="xingkong_2000",
        config={
            "model": "Qwen/Qwen3-8B",
            "dataset": "linxy/LaTeX_OCR",
            "prompt": "不知道",
            "train_data_number": 2000,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_RANK*2,
            "lora_dropout": 0.0,
        },
    )

    # -------------------------
    # SFT阶段
    # -------------------------
    sft_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    sft_dataset = sft_dataset.shuffle(seed=SEED).select(range(5))
    sft_dataset = format_sft_dataset(sft_dataset, tokenizer)
    train_sft(model, tokenizer, sft_dataset, swanlab_callback=swanlab_callback)

    # -------------------------
    # GRPO阶段
    # -------------------------
    del sft_dataset
    clean_memory()

    grpo_dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    grpo_dataset = grpo_dataset.shuffle(seed=SEED).select(range(5))

    grpo_dataset = grpo_dataset.map(lambda x: {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["prompt"]}],
        "answer": x["solution"]
    })
    max_prompt_length = 256
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
    train_grpo(model, tokenizer, grpo_dataset, max_prompt_length, max_completion_length, swanlab_callback=swanlab_callback)

    # -------------------------
    # 保存LoRA
    # -------------------------
    model.save_pretrained("grpo_saved_lora")

if __name__ == "__main__":
    main()