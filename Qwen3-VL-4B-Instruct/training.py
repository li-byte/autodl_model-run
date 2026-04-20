import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

if __name__ == '__main__':

    # -----------------------------
    # 模型与 Processor
    # -----------------------------
    model_name = "../../models/Qwen3-VL-8B-Instruct"

    # 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )

    # 加载 Processor
    processor = AutoProcessor.from_pretrained(model_name)

    # -----------------------------
    # LoRA 配置（加大版）
    # -----------------------------
    lora_config = LoraConfig(
        r=128,  # 更高的低秩矩阵 rank，提升学习能力
        lora_alpha=256,  # 更大权重缩放，让微调样本影响更显著
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,  # 适度增加 dropout，防止过拟合
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # -----------------------------
    # 数据准备
    # -----------------------------
    dataset = load_dataset("json", data_files="train_augmented.jsonl")["train"]

    # -----------------------------
    # tokenize 函数
    # -----------------------------
    def tokenize(batch):
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for instruction, input_dict, output_text in zip(batch["instruction"], batch["input"], batch["output"]):
            text_input = f"{instruction}\n{input_dict['text']}"
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": input_dict["image"]},
                    {"type": "text", "text": text_input},
                ]}
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            input_ids_list.append(inputs.input_ids.squeeze(0))
            attention_mask_list.append(inputs.attention_mask.squeeze(0))

            labels = processor.tokenizer(
                output_text,
                truncation=True,
                max_length=inputs.input_ids.size(1),
                padding="max_length",
                return_tensors="pt"
            ).input_ids.squeeze(0)
            labels_list.append(labels)

        input_ids_padded = pad_sequence(input_ids_list, batch_first=True,
                                        padding_value=processor.tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # -----------------------------
    # 训练参数（强化版）
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="./qwen3-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # 模拟更大 batch
        learning_rate=5e-5,  # 更小学习率，避免破坏原模型
        num_train_epochs=20,  # 多轮训练，让模型充分学习
        fp16=True,
        save_strategy="epoch",
        logging_steps=5,
        save_total_limit=5
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # -----------------------------
    # 开始训练
    # -----------------------------
    trainer.train()

    # -----------------------------
    # 保存 LoRA 权重
    # -----------------------------
    model.save_pretrained("./qwen3-lora-final")