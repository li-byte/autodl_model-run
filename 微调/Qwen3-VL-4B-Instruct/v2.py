import glob
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from PIL import Image
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments, Trainer
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration


# ---------------------------
# 数据整理器
# ---------------------------
class Qwen3VLDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_id_tensors = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_tensors = [torch.as_tensor(f["attention_mask"], dtype=torch.long) for f in features]
        label_tensors = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]

        max_length = max(t.size(0) for t in input_id_tensors)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("pad_token_id 与 eos_token_id 均为 None，无法进行 padding。")

        input_ids = torch.full((len(features), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)

        for idx, (ids, attn, lbl) in enumerate(zip(input_id_tensors, attention_tensors, label_tensors)):
            length = ids.size(0)
            input_ids[idx, :length] = ids
            attention_mask[idx, :length] = attn
            labels[idx, :length] = lbl

        pixel_tensors = []
        for sample in features:
            pv = sample["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_tensors.append(pv)
        pixel_values = torch.cat(pixel_tensors, dim=0)

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
# prompt
# ---------------------------
PROMPT_TEXT = "转录此图像的LaTeX."

# ---------------------------
# 本地数据加载函数
# ---------------------------
def load_local_dataset(data_dir: str, file_ext="json") -> List[Dict[str, Any]]:


    dataset = []
    files = glob.glob(os.path.join(data_dir, f"*.{file_ext}"))
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                # 自动拼接绝对路径
                img_path = os.path.join(data_dir, item["image_path"])
                img = Image.open(img_path).convert("RGB")
                dataset.append({
                    "image": img,
                    "text": item["text"]
                })
    return dataset

# ---------------------------
# 样本处理
# ---------------------------
def process_func(example, tokenizer, processor):
    MAX_LENGTH = 8192
    image = example["image"]
    output_content = example["text"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,
    )

    instruction_input_ids = inputs["input_ids"][0]
    instruction_attention_mask = inputs["attention_mask"][0]
    instruction_pixel_values = inputs["pixel_values"]
    instruction_image_grid_thw = inputs["image_grid_thw"][0]

    response = tokenizer(f"{output_content}", add_special_tokens=False)
    response_input_ids = response["input_ids"]
    response_attention_mask = response.get("attention_mask", [1] * len(response_input_ids))

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

    input_ids = instruction_input_ids + response_input_ids
    attention_mask = instruction_attention_mask + response_attention_mask
    labels = [-100] * len(instruction_input_ids) + response_input_ids

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
    load_dotenv()
    os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj"

    # 本地数据路径
    train_dir = "data"
    test_dir = "data"

    train_data = load_local_dataset(train_dir)
    test_data = load_local_dataset(test_dir)
    print(f"训练样本数量: {len(train_data)}, 测试样本数量: {len(test_data)}")

    model_id = "../../models/Qwen3-VL-4B-Instruct"
    output_dir = "./Qwen3-VL-4B-local"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )
    model.to(dtype=torch.bfloat16)
    model.config.use_cache = False

    map_kwargs = {"tokenizer": tokenizer, "processor": processor}
    train_dataset = [process_func(sample, **map_kwargs) for sample in train_data]
    eval_dataset = [process_func(sample, **map_kwargs) for sample in test_data]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        r=128,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.enable_input_require_grads()

    swanlab_callback = SwanLabCallback(
        project="Qwen3-VL-finetune",
        experiment_name="xingkong_2000",
        config={
            "model": model_id,
            "dataset": "local_data",
            "prompt": PROMPT_TEXT,
            "train_data_number": len(train_data),
        },
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        logging_first_step=5,
        num_train_epochs=8,
        save_steps=50,
        save_total_limit=3,
        learning_rate=1e-4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Qwen3VLDataCollator(tokenizer=tokenizer),
        callbacks=[swanlab_callback],
    )

    trainer.train()

    # 绘制损失曲线
    logs = trainer.state.log_history
    steps = [log['step'] for log in logs if 'loss' in log]
    losses = [log['loss'] for log in logs if 'loss' in log]
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (Local Data)')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()