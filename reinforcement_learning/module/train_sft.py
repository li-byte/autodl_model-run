# train_sft.py
from trl import SFTTrainer, SFTConfig
from config import SEED

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
            num_train_epochs=1,
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