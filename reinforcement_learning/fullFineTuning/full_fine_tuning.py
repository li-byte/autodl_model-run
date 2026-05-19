# main_full_finetune_optimized.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 减少显存碎片

import re
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# -----------------------------
# 配置参数
# -----------------------------
MAX_SEQ_LENGTH = 2048
SEED = 3407

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
# 本地数据集
# -----------------------------
def create_local_dataset(tokenizer):
    """
    创建本地数据集
    """
    data = [
        {
            "problem": "小明喜欢写什么代码？",
            "thoughts": "根据已知信息，小明喜欢写Java代码。这是一个直接的个人偏好问题。",
            "answer": "Java"
        },
        {
            "problem": "小明不喜欢写什么代码？",
            "thoughts": "已知小明不喜欢写Python，这是一个明确的偏好信息。",
            "answer": "Python"
        },
        {
            "problem": "小明的职业是什么？",
            "thoughts": "根据已知信息，小明是一位Java架构师。这是对职业的直接描述。",
            "answer": "Java架构师"
        },
        {
            "problem": "如果小明是Java架构师，他喜欢写什么代码？",
            "thoughts": "作为Java架构师，小明主要使用Java技术栈，且已知他喜欢写Java代码。因此他喜欢写Java代码。",
            "answer": "Java"
        }
    ]

    df = pd.DataFrame(data)

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

    dataset = Dataset.from_pandas(df)
    print(f"✅ 本地数据集创建完成，包含 {len(dataset)} 个样本")

    # 对数据集进行tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


# -----------------------------
# 初始化模型（优化显存使用）
# -----------------------------
def init_model():
    """
    初始化基础语言模型用于全量微调 - 优化显存版本
    """
    # 使用更保守的设备映射
    model = AutoModelForCausalLM.from_pretrained(
        "../../../models/Qwen3-4B",  # 修改为您的模型路径
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # 禁用KV cache，减少显存
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "../../../models/Qwen3-4B",
        trust_remote_code=True,
    )

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 启用梯度检查点
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型总参数: {total_params:,}")
    print(f"📊 可训练参数: {trainable_params:,}")
    print(f"📊 全量微调占比: {trainable_params / total_params * 100:.2f}%")

    # 打印显存使用情况
    if torch.cuda.is_available():
        print(f"💾 初始显存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    return model, tokenizer


# -----------------------------
# 全量微调训练（优化版）
# -----------------------------
def train_full_finetune(model, tokenizer, dataset):
    """2
    使用全量微调训练模型 - 优化显存版本
    """
    print("\n🚀 开始全量微调训练...")

    # 清理显存
    torch.cuda.empty_cache()

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 训练参数 - 针对显存优化
    training_args = TrainingArguments(
        output_dir="./full_finetune_output",
        # 批次大小优化
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 增加到8，保持有效批次大小
        # 训练轮数
        num_train_epochs=100,
        # 优化器设置
        optim="adamw_8bit",  # 使用8-bit优化器显著减少显存
        learning_rate=2e-5,
        # 梯度检查点
        gradient_checkpointing=True,
        # 混合精度
        fp16=False,  # 如果GPU支持fp16，可以改为True
        bf16=True,
        # 其他设置
        warmup_steps=5,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,  # 只保留最后一个checkpoint
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
        # 显存优化关键参数
        remove_unused_columns=False,  # 避免额外的显存开销
        ddp_find_unused_parameters=False,  # 如果是多GPU
        group_by_length=False,  # 避免重新排序增加显存
        length_column_name="length",
        # 预测时禁用（训练不需要）
        prediction_loss_only=True,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 打印训练前的显存
    if torch.cuda.is_available():
        print(f"💾 训练前显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"💾 显存缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    # 开始训练
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ 显存不足")
        raise e

    print("✅ 全量微调训练完成")
    return model


# -----------------------------
# 推理测试函数
# -----------------------------
def test_inference(model, tokenizer, problem):
    """
    测试模型推理能力
    """
    print(f"\n🧪 测试问题: {problem}")

    model.eval()  # 设置为评估模式

    # 构建prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # 生成回答
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # 推理时重新启用cache
        )

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # 提取答案
    answer_match = re.search(rf"{SOLUTION_START}(.+?){SOLUTION_END}", output_text, re.DOTALL)
    reasoning_match = re.search(rf"{REASONING_START}(.+?){REASONING_END}", output_text, re.DOTALL)

    print(f"💭 推理过程: {reasoning_match.group(1) if reasoning_match else '未找到'}")
    print(f"📝 最终答案: {answer_match.group(1) if answer_match else '未找到'}")
    print(f"📄 完整输出: {output_text[:500]}...")

    return output_text


# -----------------------------
# 主函数
# -----------------------------
def main():
    print("=" * 50)
    print("🎯 开始全量微调训练（显存优化版）")
    print("=" * 50)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"🖥️  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"💾 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 初始化模型
    print("\n📦 初始化模型...")
    model, tokenizer = init_model()

    # 创建本地数据集
    print("\n📊 创建本地数据集...")
    dataset = create_local_dataset(tokenizer)
    print(f"📊 训练样本数: {len(dataset)}")

    # 全量微调训练
    model = train_full_finetune(model, tokenizer, dataset)

    # 保存最终模型
    print("\n💾 保存最终模型...")
    model.save_pretrained("./full_finetuned_model")
    tokenizer.save_pretrained("./full_finetuned_model")
    print("✅ 模型已保存到 ./full_finetuned_model")

    # 测试推理（可选）
    print("\n🔍 测试推理...")
    test_inference(model, tokenizer, "陈嘉宇喜欢写什么代码？")

    print("\n🎉 全量微调完成！")


if __name__ == "__main__":
    main()