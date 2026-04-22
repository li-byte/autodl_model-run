from trl import SFTTrainer, SFTConfig
from config import SEED
import torch


def train_sft(model, tokenizer, dataset, swanlab_callback=None):
    """
    使用监督微调（SFT）对模型进行训练。

    参数:
        model: 待训练模型（通常包含LoRA）
        tokenizer: 模型对应的分词器
        dataset: 已格式化SFT训练数据集，包含 'text' 字段
        swanlab_callback: SwanLab实验追踪回调，可选

    返回:
        None, 训练过程会直接更新模型参数
    """

    # -----------------------------
    # 1. 初始化SFT训练器
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",  # 数据集中用于训练的文本字段
            per_device_train_batch_size=1,  # 每个GPU的训练batch size
            gradient_accumulation_steps=1,  # 梯度累积步数，用于小显存训练大batch
            warmup_steps=5,  # 学习率预热步数
            num_train_epochs=1,  # 总训练轮数
            learning_rate=2e-4,  # 学习率
            logging_steps=5,  # 每多少步打印一次日志
            optim="adamw_8bit",  # 使用8bit AdamW优化器，节省显存
            weight_decay=0.01,  # 权重衰减
            lr_scheduler_type="linear",  # 学习率调度策略
            seed=SEED,  # 随机种子，保证可复现
            report_to="swanlab" if swanlab_callback else None,  # 日志报告到SwanLab
        ),
        callbacks=[swanlab_callback] if swanlab_callback else None  # 训练回调，可用于日志记录或早停
    )

    # -----------------------------
    # 2. 启动训练
    # -----------------------------
    trainer.train()