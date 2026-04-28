# model_utils.py
import os
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from config import MAX_SEQ_LENGTH, LORA_RANK, GPU_MEMORY_UTILIZATION, SEED

# -----------------------------
# 初始化模型和LoRA
# -----------------------------
def init_model(load_lora_path=None, for_training=True):
    """
    初始化基础语言模型并可选加载LoRA适配层
    """
    # 加载基础模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        "../../../models/Qwen3-4B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    if load_lora_path and os.path.exists(load_lora_path):
        print(f"加载已有LoRA: {load_lora_path}")
        # 先转换为PEFT模型
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK * 2,
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
        # 加载已保存的权重
        model = PeftModel.from_pretrained(model, load_lora_path)
        if for_training:
            model.train()  # 设置为训练模式
            # 确保所有参数都正确设置梯度
            for param in model.parameters():
                param.requires_grad = True
    else:
        # 使用新LoRA进行低秩适配
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK * 2,
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

    return model, tokenizer