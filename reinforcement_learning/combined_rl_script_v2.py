# main_local.py
# ============================================================================
# 导入必要的库
# ============================================================================
import os

# 设置HuggingFace镜像源，解决国内网络访问问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用HF传输加速（避免某些环境下的兼容性问题）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import gc
import re
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from vllm import SamplingParams
from swanlab.integration.transformers import SwanLabCallback
from peft import PeftModel

# ============================================================================
# 配置参数（全局常量）
# ============================================================================
MAX_SEQ_LENGTH = 2048
LORA_RANK = 4
SEED = 3407
GPU_MEMORY_UTILIZATION = 0.7

# ============================================================================
# 系统提示和答案标记
# ============================================================================
REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"

SYSTEM_PROMPT = f"""你将得到一个问题。
请仔细思考并给出你的推理过程。
将推理过程放在 {REASONING_START} 和 {REASONING_END} 之间。
然后，将最终答案放在 {SOLUTION_START} 和 {SOLUTION_END} 之间。"""


# ============================================================================
# 函数：初始化模型和LoRA适配层
# ============================================================================
def init_model(load_lora_path=None, for_training=True):
    """
    初始化基础语言模型并可选加载LoRA适配层
    """
    # 加载基础模型（请修改为你的本地模型路径）
    model, tokenizer = FastLanguageModel.from_pretrained(
        "../../models/Qwen3-4B",  # 请修改为你的模型路径
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    # 检查是否需要加载已有的LoRA权重
    if load_lora_path and os.path.exists(load_lora_path):
        print(f"加载已有LoRA: {load_lora_path}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK * 2,
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
        model = PeftModel.from_pretrained(model, load_lora_path)
        if for_training:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
    else:
        # 没有已有LoRA，创建新的LoRA适配层
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


# ============================================================================
# 函数：创建本地SFT数据集
# ============================================================================
def create_local_sft_dataset(tokenizer):
    """
    创建本地SFT（监督微调）数据集

    数据格式: 包含问题、推理过程和答案
    """
    # 本地数据：3个简单的数学问题
    local_data = [
        {
            "problem": "小明有5个苹果，给了小红2个，还剩几个？",
            "generated_solution": "小明最初有5个苹果，给了小红2个，所以需要减去2个。5 - 2 = 3。",
            "expected_answer": "3"
        },
        {
            "problem": "一个长方形的长是8米，宽是5米，它的面积是多少？",
            "generated_solution": "长方形面积 = 长 × 宽 = 8 × 5 = 40 平方米。",
            "expected_answer": "40"
        },
        {
            "problem": "一包糖果有24颗，平均分给6个小朋友，每人分到几颗？",
            "generated_solution": "24颗糖果分给6个小朋友，每人数量 = 24 ÷ 6 = 4 颗。",
            "expected_answer": "4"
        }
    ]

    # 转换为DataFrame
    df = pd.DataFrame(local_data)

    # 格式化每条数据
    formatted_messages = []
    for _, row in df.iterrows():
        # 清理推理过程
        thoughts = row["generated_solution"].strip()

        # 构建最终的助手回复内容
        final_prompt = (REASONING_START + thoughts + REASONING_END +
                        SOLUTION_START + row["expected_answer"] + SOLUTION_END)

        # 构建消息列表
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": final_prompt},
        ]
        formatted_messages.append(messages)

    # 应用聊天模板
    df["Messages"] = formatted_messages
    df["text"] = tokenizer.apply_chat_template(
        df["Messages"].values.tolist(),
        tokenize=False
    )

    # 返回HuggingFace Dataset
    return Dataset.from_pandas(df[["text"]])


# ============================================================================
# 函数：创建本地GRPO数据集
# ============================================================================
def create_local_grpo_dataset():
    """
    创建本地GRPO（强化学习）数据集

    数据格式: 包含提示问题和答案
    """
    # 本地数据：3个简单的数学问题
    local_data = [
        {
            "prompt": "如果每支铅笔2元，买5支铅笔需要多少钱？",
            "solution": "10"
        },
        {
            "prompt": "一个三角形有三条边，分别是3cm、4cm、5cm，它的周长是多少？",
            "solution": "12"
        },
        {
            "prompt": "小华有15元，买了一本书花了8元，还剩多少钱？",
            "solution": "7"
        }
    ]

    # 转换为GRPO需要的格式
    formatted_data = []
    for item in local_data:
        formatted_data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]}
            ],
            "answer": item["solution"]
        })

    # 转换为Dataset
    df = pd.DataFrame(formatted_data)
    return Dataset.from_pandas(df)


# ============================================================================
# 函数：监督微调（SFT）训练
# ============================================================================
def train_sft(model, tokenizer, dataset, swanlab_callback=None):
    """
    使用监督微调训练模型
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
            num_train_epochs=3,  # 小数据集可以多训练几轮
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            report_to="swanlab" if swanlab_callback else None,
            output_dir="./sft_output",  # 添加输出目录
        ),
        callbacks=[swanlab_callback] if swanlab_callback else None
    )

    trainer.train()
    print("SFT训练完成！")


# ============================================================================
# 奖励函数1：检查输出格式
# ============================================================================
def match_format_approximately(completions, **kwargs):
    """
    GRPO奖励函数：检查格式是否正确
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0

        # 检查标记出现次数
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0

        scores.append(score)
    return scores


# ============================================================================
# 奖励函数2：检查答案是否正确
# ============================================================================
def check_answer(prompts, completions, answer, **kwargs):
    """
    GRPO奖励函数：检查答案是否正确
    """
    match_format = re.compile(
        rf"{REASONING_END}.*?{SOLUTION_START}(.+?){SOLUTION_END}.*$",
        flags=re.MULTILINE | re.DOTALL
    )

    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for response, true_answer in zip(responses, answer):
        guess = match_format.search(response)

        if guess is None:
            scores.append(-2.0)
        else:
            extracted_answer = guess.group(1).strip()
            if extracted_answer == true_answer.strip():
                scores.append(5.0)
            else:
                scores.append(-1.5)

    return scores


# ============================================================================
# 函数：GRPO训练
# ============================================================================
def train_grpo(model, tokenizer, dataset, max_prompt_length, max_completion_length, swanlab_callback=None):
    """
    使用GRPO进行强化学习训练
    """
    sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=SEED,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True
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
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=10,
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
    print("GRPO训练完成！")


# ============================================================================
# 函数：测试模型
# ============================================================================
def test_model(model, tokenizer):
    """
    测试训练后的模型
    """
    test_questions = [
        "小明有10个苹果，送给小红3个，还剩几个？",
        "一个正方形边长是6米，它的面积是多少？",
    ]

    print("\n" + "=" * 50)
    print("开始测试模型")
    print("=" * 50)

    for question in test_questions:
        # 构建完整的提示
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        # 应用聊天模板
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        # 生成回答
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"\n问题: {question}")
        print(f"回答: {response}")
        print("-" * 50)


# ============================================================================
# 主函数
# ============================================================================
def main():
    """
    主执行流程：
    1. 初始化模型
    2. 准备本地数据
    3. SFT训练
    4. GRPO训练
    5. 测试模型
    6. 保存模型
    """

    # -------------------------
    # 步骤1: 初始化模型
    # -------------------------
    print("正在初始化模型...")
    model, tokenizer = init_model(load_lora_path=None)

    # -------------------------
    # 步骤2: 配置 SwanLab（可选）
    # -------------------------
    # 如果没有SwanLab账号，可以注释掉这部分
    try:
        os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj" # 替换为你的API key
        swanlab_callback = SwanLabCallback(
            project="Qwen3-Local-Finetune",
            experiment_name="local_test",
            config={
                "model": "Qwen3-4B",
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_RANK * 2,
                "training_type": "SFT+GRPO",
            },
        )
        print("SwanLab已启用")
    except:
        swanlab_callback = None
        print("SwanLab未配置，将跳过实验追踪")

    # -------------------------
    # 步骤3: 准备SFT数据集
    # -------------------------
    print("\n准备SFT数据集...")
    sft_dataset = create_local_sft_dataset(tokenizer)
    print(f"SFT数据集大小: {len(sft_dataset)}")
    print("示例数据:")
    print(sft_dataset[0]["text"][:200] + "...")

    # -------------------------
    # 步骤4: SFT训练
    # -------------------------
    print("\n开始SFT训练...")
    train_sft(model, tokenizer, sft_dataset, swanlab_callback=swanlab_callback)

    # -------------------------
    # 步骤5: 清理显存
    # -------------------------
    del sft_dataset
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------
    # 步骤6: 准备GRPO数据集
    # -------------------------
    print("\n准备GRPO数据集...")
    grpo_dataset = create_local_grpo_dataset()
    print(f"GRPO数据集大小: {len(grpo_dataset)}")
    print("示例数据:")
    print(grpo_dataset[0])

    # 计算长度限制
    max_prompt_length = 256
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length

    # -------------------------
    # 步骤7: GRPO训练
    # -------------------------
    print("\n开始GRPO训练...")
    train_grpo(model, tokenizer, grpo_dataset,
               max_prompt_length, max_completion_length,
               swanlab_callback=swanlab_callback)

    # -------------------------
    # 步骤8: 测试模型
    # -------------------------
    test_model(model, tokenizer)

    # -------------------------
    # 步骤9: 保存模型
    # -------------------------
    print("\n保存模型...")
    model.save_pretrained("local_finetuned_lora")
    tokenizer.save_pretrained("local_finetuned_lora")
    print("模型已保存到 local_finetuned_lora 目录")


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    main()