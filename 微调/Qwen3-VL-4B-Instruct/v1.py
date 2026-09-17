
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
from typing import Any, Dict, List
# 数据集加载
from datasets import load_dataset

# 自定义视觉处理工具函数
from qwen_vl_utils import process_vision_info

# PEFT (Parameter-Efficient Fine-Tuning) 相关工具
from peft import LoraConfig, TaskType, get_peft_model

# Transformers 相关库
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
)

# 用于动态导入模块
import importlib

# 绘图库，用于绘制训练损失曲线
import matplotlib.pyplot as plt

# SwanLab 回调，用于实验追踪
from swanlab.integration.transformers import SwanLabCallback

# 加载环境变量
from dotenv import load_dotenv


# ---------------------------
# 自定义数据整理器
# ---------------------------
class Qwen3VLDataCollator:
    """
    数据整理器，将处理后的样本整理成 Trainer 可接受的 batch。
    包含：
    - 文本输入 (input_ids, attention_mask, labels)
    - 图像输入 (pixel_values)
    - 图像网格信息 (image_grid_thw)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 将每个样本的 input_ids, attention_mask, labels 转为 tensor
        input_id_tensors = [torch.as_tensor(sample["input_ids"], dtype=torch.long) for sample in features]
        attention_tensors = [torch.as_tensor(sample["attention_mask"], dtype=torch.long) for sample in features]
        label_tensors = [torch.as_tensor(sample["labels"], dtype=torch.long) for sample in features]

        # 计算 batch 中的最大长度，用于 padding
        max_length = max(t.size(0) for t in input_id_tensors)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("pad_token_id 与 eos_token_id 均为 None，无法进行 padding。")

        # 初始化 padding 后的 tensor
        input_ids = torch.full((len(features), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)  # -100 表示忽略位置

        # 填充实际数据
        for idx, (ids, attn, lbl) in enumerate(zip(input_id_tensors, attention_tensors, label_tensors)):
            length = ids.size(0)
            input_ids[idx, :length] = ids
            attention_mask[idx, :length] = attn
            labels[idx, :length] = lbl

        # 处理图像输入
        pixel_tensors = []
        for sample in features:
            pv = sample["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_tensors.append(pv)
        pixel_values = torch.cat(pixel_tensors, dim=0)

        # 处理图像网格信息
        image_grid_thw = torch.stack(
            [torch.as_tensor(sample["image_grid_thw"], dtype=torch.long).view(-1) for sample in features], dim=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


# ---------------------------
# 训练 prompt
# ---------------------------
PROMPT_TEXT = "转录此图像的LaTeX."


# ---------------------------
# 数据预处理函数
# ---------------------------
def process_func(example, tokenizer, processor):
    """
    将单条样本处理成模型可接受的格式
    """
    MAX_LENGTH = 8192  # 最大序列长度

    image = example["image"]  # 图像数据
    output_content = example["text"]  # 对应 LaTeX 文本

    # 构造消息列表，包含用户图像和提示
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    # 使用 processor 构造文本输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 使用自定义函数处理视觉信息
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,
    )

    # 获取 instruction 部分的输入
    instruction_input_ids = inputs["input_ids"][0]
    instruction_attention_mask = inputs["attention_mask"][0]
    instruction_pixel_values = inputs["pixel_values"]
    instruction_image_grid_thw = inputs["image_grid_thw"][0]

    # 处理响应部分 (LaTeX 输出)
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    response_input_ids = response["input_ids"]
    response_attention_mask = response.get("attention_mask", [1] * len(response_input_ids))

    # 添加 eos_token_id 或 pad_token_id 结束响应
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        if not response_input_ids or response_input_ids[-1] != eos_token_id:
            response_input_ids.append(eos_token_id)
            response_attention_mask.append(1)
    else:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("需要定义 eos_token_id 或 pad_token_id 才能结束响应序列。")
        response_input_ids.append(pad_token_id)
        response_attention_mask.append(1)

    # 拼接 instruction 和 response
    input_ids = instruction_input_ids + response_input_ids
    attention_mask = instruction_attention_mask + response_attention_mask
    labels = [-100] * len(instruction_input_ids) + response_input_ids

    # 截断到最大长度
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": instruction_pixel_values,
        "image_grid_thw": instruction_image_grid_thw,
    }


# ---------------------------
# 主函数
# ---------------------------
def main():
    # 加载环境变量
    load_dotenv()
    # SWANLAB_API_KEY 请通过环境变量配置（.env 或系统环境变量），勿硬编码在代码中

    # 数据采样比例
    data_fraction = 0.002

    # 加载数据集
    ds = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")
    ds = ds.shuffle(seed=222)

    # 训练/测试数据采样
    train_data = ds["train"].select(range(int(len(ds["train"]) * data_fraction)))
    print(f"训练数据大小: {len(train_data)}")
    test_data = ds["test"].select(range(int(len(ds["test"]) * data_fraction)))
    print(f"测试数据大小: {len(test_data)}")

    # 模型路径与输出目录
    model_id = "../../models/Qwen3-VL-4B-Instruct"
    output_dir = "./Qwen3-VL-4B"

    # 加载 tokenizer 和 processor
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False,
                                              trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    # 加载模型
    # 动态加载模型
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    module_name = f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    module = importlib.import_module(module_name)
    model_cls = getattr(module, arch)
    model = model_cls.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )

    # 转换模型 dtype
    model.to(dtype=torch.bfloat16)
    model.config.use_cache = False

    # 处理训练数据
    map_kwargs = {"tokenizer": tokenizer, "processor": processor}
    train_dataset = train_data.map(
        process_func,
        remove_columns=train_data.column_names,
        fn_kwargs=map_kwargs,
    )

    # LoRA 配置
    lora_config_dict = {
        "lora_rank": 128,
        "lora_alpha": 16,
        "lora_dropout": 0,
    }
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=lora_config_dict["lora_rank"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias="none",
    )

    # 获取 PEFT 模型
    peft_model = get_peft_model(model, config)
    peft_model.enable_input_require_grads()
    # SWANLAB_API_KEY 请通过环境变量配置（.env 或系统环境变量），勿硬编码在代码中
    # SwanLab 回调配置
    swanlab_callback = SwanLabCallback(
        project="Qwen3-VL-finetune",
        experiment_name="experiment_2000",
        config={
            "model": model_id,
            "dataset": "linxy/LaTeX_OCR",
            "prompt": PROMPT_TEXT,
            "train_data_number": len(train_data),
            "lora_rank": lora_config_dict["lora_rank"],
            "lora_alpha": lora_config_dict["lora_alpha"],
            "lora_dropout": lora_config_dict["lora_dropout"],
        },
    )

    # 训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # 每个 GPU batch size
        gradient_accumulation_steps=4,  # 梯度累积步数
        logging_steps=10,
        logging_first_step=5,
        num_train_epochs=8,  # 训练轮数
        save_steps=50,  # 保存模型间隔
        save_total_limit=3,  # 最大保存模型数量
        learning_rate=1e-4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    # 处理评估数据
    eval_dataset = test_data.map(
        process_func,
        remove_columns=test_data.column_names,
        fn_kwargs=map_kwargs,
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Qwen3VLDataCollator(tokenizer=tokenizer),
        callbacks=[swanlab_callback],
    )

    # 开始训练
    trainer.train()

    # 绘制训练损失曲线
    logs = trainer.state.log_history
    steps = [log['step'] for log in logs if 'loss' in log]
    losses = [log['loss'] for log in logs if 'loss' in log]
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (Qwen3-VL-30B)')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))

    # 保存模型与 tokenizer
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()