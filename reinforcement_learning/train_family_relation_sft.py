# main.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj"
import gc
import re
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from swanlab.integration.transformers import SwanLabCallback
from peft import PeftModel

# -----------------------------
# 配置参数
# -----------------------------
MAX_SEQ_LENGTH = 2048
LORA_RANK = 4
SEED = 3407
GPU_MEMORY_UTILIZATION = 0.7

# -----------------------------
# 系统提示和答案标记
# -----------------------------
REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"

SYSTEM_PROMPT = f"""你将得到一个问题。
请仔细思考并给出你的推理过程。
将推理过程放在 {REASONING_START} 和 {REASONING_END} 之间。
然后，将最终答案放在 {SOLUTION_START} 和 {SOLUTION_END} 之间。"""


# -----------------------------
# 本地数据集 - 小明的爸爸是小林
# -----------------------------
def create_local_dataset(tokenizer):
    """
    创建本地数据集："小明的爸爸是小林" 的关系推理
    """

    data = [
    {
        "problem": "陈嘉宇的叔叔是谁？",
        "thoughts": "根据家庭关系信息，陈嘉宇的叔叔是小明。",
        "answer": "小明"
    },
    {
        "problem": "陈嘉宇周末要跟小明去爬山？",
        "thoughts": "根据已知信息，陈嘉宇周末要跟小明去梧桐山爬山。",
        "answer": "是的，去梧桐山"
    },
    {
        "problem": "陈嘉宇的叔叔叫什么名字？",
        "thoughts": "根据家庭关系信息，陈嘉宇的叔叔是小明。",
        "answer": "小明"
    }
]

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 格式化数据为聊天格式
    def format_row(x):
        final_prompt = REASONING_START + x["thoughts"] + REASONING_END + SOLUTION_START + x["answer"] + SOLUTION_END
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["problem"]},
            {"role": "assistant", "content": final_prompt},
        ]
        return messages

    df["Messages"] = df.apply(format_row, axis=1)
    df["text"] = tokenizer.apply_chat_template(df["Messages"].values.tolist(), tokenize=False)

    # 创建Dataset对象
    dataset = Dataset.from_pandas(df)
    print(f"✅ 本地数据集创建完成，包含 {len(dataset)} 个样本")
    print("📝 样本示例:")
    for i, row in enumerate(df.itertuples()):
        print(f"  {i + 1}. 问题: {row.problem} -> 答案: {row.answer}")

    return dataset


# -----------------------------
# 初始化模型和LoRA
# -----------------------------
def init_model():
    """
    初始化基础语言模型并可选加载LoRA适配层
    """
    # 加载基础模型（请根据实际路径修改）
    model, tokenizer = FastLanguageModel.from_pretrained(
        "../../models/Qwen3-4B",  # 修改为您的模型路径
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
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
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    return model, tokenizer


# -----------------------------
# SFT训练（简化版）
# -----------------------------
def train_sft(model, tokenizer, dataset, swanlab_callback=None):
    """
    使用监督微调（SFT）训练模型
    """
    print("\n🚀 开始SFT训练...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=100,  # 小数据集增加训练轮数
            learning_rate=2e-4,
            logging_steps=5,
            save_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir="sft_output",
            report_to="swanlab" if swanlab_callback else None,
        ),
        callbacks=[swanlab_callback] if swanlab_callback else None
    )

    trainer.train()
    print("✅ SFT训练完成")
    #
    # # 保存SFT后的模型
    # model.save_pretrained("sft_saved_lora")
    # print("💾 LoRA已保存到 sft_saved_lora")

    return model


# -----------------------------
# 推理测试函数
# -----------------------------
def test_inference(model, tokenizer, problem):
    """
    测试模型推理能力
    """
    print(f"\n🧪 测试问题: {problem}")

    # 构建prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # 生成回答
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    output_tokens = model.fast_generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # 提取答案
    answer_match = re.search(rf"{SOLUTION_START}(.+?){SOLUTION_END}", output_text, re.DOTALL)
    reasoning_match = re.search(rf"{REASONING_START}(.+?){REASONING_END}", output_text, re.DOTALL)

    print(f"💭 推理过程: {reasoning_match.group(1) if reasoning_match else '未找到'}")
    print(f"📝 最终答案: {answer_match.group(1) if answer_match else '未找到'}")
    print(f"📄 完整输出: {output_text}")

    return output_text


# -----------------------------
# 主函数
# -----------------------------
def main():
    print("=" * 50)
    print("🎯 开始训练 '小明的爸爸是小林' 关系推理模型")
    print("=" * 50)

    # 初始化模型
    print("\n📦 初始化模型...")
    model, tokenizer = init_model()

    # 创建本地数据集
    print("\n📊 创建本地数据集...")
    dataset = create_local_dataset(tokenizer)

    # -------------------------
    # 配置 SwanLab 回调（可选）
    # -------------------------
    swanlab_callback = None

    # -------------------------
    # SFT训练
    # -------------------------
    model = train_sft(model, tokenizer, dataset, swanlab_callback=swanlab_callback)
    # -------------------------
    # 保存最终模型
    # -------------------------
    print("\n💾 保存最终模型...")
    # 编程
    # model.save_pretrained("programming_lora")
    # print("✅ 模型已保存到 programming_lora")
    #家庭
    model.save_pretrained("family_relation_lora")
    print("✅ 模型已保存到 family_relation_lora")

    print("\n🎉 训练完成！")


if __name__ == "__main__":
    main()
