import torch
from unsloth import FastLanguageModel
from config import LORA_RANK, MAX_SEQ_LENGTH, GPU_MEMORY_UTILIZATION, SEED


def init_model(checkpoint_path=None):
    """
    初始化基础语言模型并加载LoRA低秩适配层。

    如果提供 checkpoint_path，则在基础模型上加载上一次训练的 LoRA 权重，实现增量训练。

    参数:
        checkpoint_path (str, optional): LoRA权重保存路径，用于增量训练。默认 None。

    返回:
        model: 已初始化并可微调的语言模型实例（包含LoRA）。
        tokenizer: 对应的分词器，用于文本编码和解码。
    """

    # -----------------------------
    # 1. 加载基础模型
    # -----------------------------
    # from_pretrained:
    # - "../../models/Qwen3-4B": 基础模型路径
    # - max_seq_length: 模型支持的最大序列长度
    # - load_in_4bit: 是否以4bit量化加载（False表示全精度）
    # - fast_inference: 是否启用快速推理优化
    # - max_lora_rank: LoRA秩的最大值
    # - gpu_memory_utilization: GPU显存占用比例上限
    model, tokenizer = FastLanguageModel.from_pretrained(
        "../../models/Qwen3-4B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    # -----------------------------
    # 2. 加载已有 LoRA 权重（增量训练）
    # -----------------------------
    # 如果用户提供 checkpoint_path，则加载上次训练保存的 LoRA 权重
    # 注意：LoRA权重必须与当前模型配置兼容（秩 r、目标模块、alpha 等）
    if checkpoint_path is not None:
        model.load_peft_model(checkpoint_path)

    # -----------------------------
    # 3. 初始化 LoRA 低秩适配层
    # -----------------------------
    # get_peft_model 会在基础模型上添加 LoRA 模块，用于微调
    # 参数说明：
    # - r: LoRA秩（低秩矩阵维度）
    # - target_modules: 需要应用LoRA的线性层列表
    # - lora_alpha: LoRA缩放系数，通常为 r*2
    # - use_gradient_checkpointing: 是否启用梯度检查点，节省显存
    # - random_state: 随机种子，保证初始化可复现
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # 返回可微调的模型和分词器
    return model, tokenizer