# main.py
import os
import gc
import re
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from vllm import SamplingParams
from transformers import TextStreamer
from safetensors import safe_open
# SwanLab 回调，用于实验追踪
from swanlab.integration.transformers import SwanLabCallback
# -----------------------------
# 配置参数
# -----------------------------
MAX_SEQ_LENGTH = 2048  # 模型输入的最大序列长度
LORA_RANK = 32  # LoRA的秩，用于低秩适配
SEED = 3407  # 随机种子，保证实验可复现
GPU_MEMORY_UTILIZATION = 0.7  # GPU显存占用比例限制

# -----------------------------
# 系统提示和答案标记
# -----------------------------
REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"

# 中文系统提示模板，指导模型输出推理过程和答案
SYSTEM_PROMPT = f"""你将得到一个问题。
请仔细思考并给出你的推理过程。
将推理过程放在 {REASONING_START} 和 {REASONING_END} 之间。
然后，将最终答案放在 {SOLUTION_START} 和 {SOLUTION_END} 之间。"""


# -----------------------------
# 初始化模型和LoRA
# -----------------------------
def init_model():
    """
    初始化基础语言模型并加载LoRA适配层
    """
    # 加载FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
       "../../models/Qwen3-4B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    # 使用LoRA进行低秩适配
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",  # 节省显存
        random_state=SEED,
    )
    return model, tokenizer


# -----------------------------
# 格式化SFT数据集
# -----------------------------
def format_sft_dataset(dataset, tokenizer):
    """
    将原始COT数据集格式化为SFT训练需要的聊天格式
    """

    def format_row(x):
        thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
        final_prompt = REASONING_START + thoughts + REASONING_END + SOLUTION_START + x["expected_answer"] + SOLUTION_END
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["problem"]},
            {"role": "assistant", "content": final_prompt},
        ]

    # 数据清洗：保留数值类型的答案
    dataset = dataset.to_pandas()
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
    dataset = dataset.iloc[np.where(is_number)[0]]

    # 格式化消息
    dataset["Messages"] = dataset.apply(format_row, axis=1)

    # 使用tokenizer生成模型输入格式
    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)
    return Dataset.from_pandas(dataset)


# -----------------------------
# SFT训练
# -----------------------------
def train_sft(model, tokenizer, dataset, swanlab_callback=None):
    """
    使用监督微调（SFT）训练模型
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            report_to="swanlab" if swanlab_callback else None,
        ),
        callbacks=[swanlab_callback] if swanlab_callback else None
    )
    trainer.train()


# -----------------------------
# GRPO奖励函数
# -----------------------------
def match_format_approximately(completions, **kwargs):
    """
    奖励函数：根据是否包含推理与答案标记，给出分数
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """
    奖励函数：根据模型答案是否正确评分
    """
    match_format = re.compile(
        rf"{REASONING_END}.*?{SOLUTION_START}(.+?){SOLUTION_END}.*$", flags=re.MULTILINE | re.DOTALL
    )
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for r, true_answer in zip(responses, answer):
        guess = match_format.search(r)
        if guess is None:
            scores.append(-2.0)
        else:
            scores.append(5.0 if guess.group(1).strip() == true_answer.strip() else -1.5)
    return scores





# -----------------------------
# GRPO训练
# -----------------------------
def train_grpo(model, tokenizer, dataset, max_prompt_length, max_completion_length, swanlab_callback=None):
    """
    使用GRPO进行强化学习训练
    """
    sampling_params = SamplingParams(
        min_p=0.1, top_p=1.0, top_k=-1, seed=SEED,
        stop=[tokenizer.eos_token], include_stop_str_in_output=True
    )
    training_args = GRPOConfig(
        vllm_sampling_params=sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=100,
        save_steps=100,
        report_to="swanlab" if swanlab_callback else None,
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[match_format_approximately, check_answer],
        args=training_args,
        train_dataset=dataset,
        callbacks=[swanlab_callback] if swanlab_callback else None
    )
    trainer.train()



# -----------------------------
# 推理示例
# -----------------------------
def inference_example(model, tokenizer, prompt):
    """
    对模型进行推理测试
    """
    sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    output = model.fast_generate([prompt], sampling_params=sampling_params)[0].outputs[0].text
    print("Inference output:", output)


# -----------------------------
# 主函数
# -----------------------------
def main():
    # 初始化模型
    model, tokenizer = init_model()

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
    sft_dataset = format_sft_dataset(sft_dataset, tokenizer)
    train_sft(model, tokenizer, sft_dataset, swanlab_callback=swanlab_callback)

    # -------------------------
    # GRPO阶段
    # -------------------------
    del sft_dataset
    torch.cuda.empty_cache()
    gc.collect()

    grpo_dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    grpo_dataset = grpo_dataset.map(lambda x: {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["prompt"]}],
        "answer": x["solution"]
    })
    max_prompt_length = 256
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
    train_grpo(model, tokenizer, grpo_dataset, max_prompt_length, max_completion_length, swanlab_callback=swanlab_callback)

    # -------------------------
    # 推理测试
    # -------------------------
    inference_example(model, tokenizer, "101的平方分数是多少？")

    # -------------------------
    # 保存LoRA
    # -------------------------
    model.save_lora("grpo_saved_lora")

if __name__ == "__main__":
    main()