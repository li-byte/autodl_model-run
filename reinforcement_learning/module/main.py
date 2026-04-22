import gc
import torch
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback

from config import MAX_SEQ_LENGTH, LORA_RANK
from dataset_utils import format_sft_dataset, format_grpo_dataset
from model_utils import init_model
from train_grpo import train_grpo
from train_sft import train_sft


def main(run_sft=True, run_grpo=True, checkpoint_path=None):
    """
    项目主函数，支持阶段性执行和增量训练

    参数:
        run_sft (bool): 是否执行 SFT 阶段
        run_grpo (bool): 是否执行 GRPO 阶段
        checkpoint_path (str or None): LoRA checkpoint路径，用于增量训练
    """

    # -------------------------
    # 1. 初始化模型
    # -------------------------
    model, tokenizer = init_model(checkpoint_path=checkpoint_path)

    # -------------------------
    # 2. 配置 SwanLab 回调
    # -------------------------
    swanlab_callback = SwanLabCallback(
        project="Qwen3-VL-finetune",
        experiment_name="xingkong_2000",
        config={
            "model": "Qwen/Qwen3-8B",
            "dataset": "linxy/LaTeX_OCR",
            "prompt": "不知道",
            "train_data_number": 2000,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_RANK * 2,
            "lora_dropout": 0.0,
        },
    )

    # -------------------------
    # 3. SFT阶段（监督微调）
    # -------------------------
    if run_sft:
        print("==> 开始 SFT 训练阶段...")
        sft_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot") \
            .shuffle(seed=3407) \
            .select(range(5))
        sft_dataset = format_sft_dataset(sft_dataset, tokenizer)

        train_sft(model, tokenizer, sft_dataset, swanlab_callback=swanlab_callback)

        # 阶段性保存 SFT 权重
        model.save_pretrained("sft_saved_lora")
        print("==> SFT 阶段权重已保存: sft_saved_lora")

        # 清理显存
        del sft_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------
    # 4. GRPO阶段（强化学习微调）
    # -------------------------
    if run_grpo:
        print("==> 开始 GRPO 训练阶段...")
        grpo_dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train") \
            .shuffle(seed=3407) \
            .select(range(5))
        grpo_dataset = format_grpo_dataset(grpo_dataset, tokenizer)

        max_prompt_length = 256
        max_completion_length = MAX_SEQ_LENGTH - max_prompt_length

        train_grpo(model, tokenizer, grpo_dataset, max_prompt_length, max_completion_length,
                   swanlab_callback=swanlab_callback)

        # 阶段性保存 GRPO 权重
        model.save_pretrained("grpo_saved_lora")
        print("==> GRPO 阶段权重已保存: grpo_saved_lora")


if __name__ == "__main__":
    # 示例: 只执行 SFT, 或只执行 GRPO, 或全部执行
    main(run_sft=True, run_grpo=True, checkpoint_path="../grpo_saved_lora")