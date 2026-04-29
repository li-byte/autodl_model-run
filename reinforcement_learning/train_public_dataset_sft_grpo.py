# main.py
# ============================================================================
# 导入必要的库
# ============================================================================
import os

# 设置HuggingFace镜像源，解决国内网络访问问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用HF传输加速（避免某些环境下的兼容性问题）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import gc  # 垃圾回收，用于清理GPU内存
import re  # 正则表达式，用于解析模型输出
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
from datasets import load_dataset, Dataset  # HuggingFace数据集加载和操作
from unsloth import FastLanguageModel  # unsloth优化的快速语言模型（比标准实现更快更省显存）
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig  # TRL库：监督微调和GRPO训练器
from vllm import SamplingParams  # vLLM推理框架的采样参数（用于GRPO生成）
from transformers import TextStreamer  # 文本流式输出（未使用，但保留导入）
from safetensors import safe_open  # 安全的张量文件加载（未直接使用）
from swanlab.integration.transformers import SwanLabCallback  # SwanLab实验追踪回调
from peft import PeftModel  # PEFT（参数高效微调）模型加载

# ============================================================================
# 配置参数（全局常量）
# ============================================================================
MAX_SEQ_LENGTH = 2048  # 模型输入的最大序列长度（token数），限制显存使用
LORA_RANK = 4  # LoRA的秩（低秩矩阵的维度），值越小参数量越少，通常4-16之间
SEED = 3407  # 随机种子，保证实验完全可复现
GPU_MEMORY_UTILIZATION = 0.7  # GPU显存占用比例限制（0.7表示最多使用70%显存）

# ============================================================================
# 系统提示和答案标记（用于结构化输出）
# ============================================================================
# 定义特殊标记，让模型输出更结构化，便于解析
REASONING_START = "<开始推理>"  # 推理过程开始标记
REASONING_END = "<结束推理>"  # 推理过程结束标记
SOLUTION_START = "<答案>"  # 答案开始标记
SOLUTION_END = "<答案结束>"  # 答案结束标记

# 中文系统提示模板，指导模型按照指定格式输出推理过程和答案
# 这是一个角色提示（system prompt），告诉模型应该如何组织输出
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

    参数:
        load_lora_path (str, optional): 已有的LoRA权重路径，如果提供则加载
        for_training (bool): 是否用于训练模式（True）还是推理模式（False）

    返回:
        model: PEFT包装的模型（支持LoRA）
        tokenizer: 对应的分词器
    """
    # 加载基础模型（Qwen3-4B）
    # from_pretrained是FastLanguageModel的类方法，专门优化加载速度
    model, tokenizer = FastLanguageModel.from_pretrained(
        "../../models/Qwen3-4B",  # 模型路径（本地已下载）
        max_seq_length=MAX_SEQ_LENGTH,  # 最大序列长度
        load_in_4bit=False,  # 是否使用4bit量化（False表示使用原始精度）
        fast_inference=False,  # 是否启用快速推理模式（False表示使用标准推理）
        max_lora_rank=LORA_RANK,  # LoRA的最大秩（用于兼容性检查）
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,  # GPU显存使用率
    )

    # 检查是否需要加载已有的LoRA权重
    if load_lora_path and os.path.exists(load_lora_path):
        print(f"加载已有LoRA: {load_lora_path}")
        # 步骤1: 创建PEFT模型结构（定义LoRA层）
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,  # LoRA的秩
            # target_modules: 哪些层应用LoRA（Qwen的注意力层和前馈层）
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK * 2,  # LoRA缩放因子（通常为r的2倍）
            use_gradient_checkpointing="unsloth",  # 梯度检查点（节省显存）
            random_state=SEED,  # 随机种子
        )
        # 步骤2: 加载已保存的LoRA权重
        model = PeftModel.from_pretrained(model, load_lora_path)
        if for_training:
            model.train()  # 设置为训练模式（启用dropout等）
            # 确保所有参数都正确设置梯度（启用自动微分）
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
# 函数：格式化SFT（监督微调）数据集
# ============================================================================
def format_sft_dataset(dataset, tokenizer):
    """
    将原始COT（思维链）数据集格式化为SFT训练需要的聊天格式

    参数:
        dataset: HuggingFace数据集，包含problem, generated_solution, expected_answer字段
        tokenizer: 分词器，用于应用聊天模板

    返回:
        Dataset: 格式化后的数据集，包含"text"字段（模型输入文本）
    """

    def format_row(x):
        """
        处理单行数据，将其转换为对话格式

        参数:
            x: 数据集的一行，包含generated_solution（推理过程）和expected_answer（答案）

        返回:
            list: 符合HuggingFace聊天模板的消息列表
        """
        # 清理推理过程：移除原始数据中的<think>标签，并去除首尾空白
        thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()

        # 构建最终的助手回复内容：推理标记+推理内容+答案标记+答案
        final_prompt = (REASONING_START + thoughts + REASONING_END +
                        SOLUTION_START + x["expected_answer"] + SOLUTION_END)

        # 返回三部分消息：系统提示、用户问题、助手回复
        return [
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统角色：设定行为规则
            {"role": "user", "content": x["problem"]},  # 用户角色：提出问题
            {"role": "assistant", "content": final_prompt},  # 助手角色：给出推理和答案
        ]

    # 转换为Pandas DataFrame以便数据处理
    dataset = dataset.to_pandas()

    # 数据清洗：只保留答案是可转换为数字的行（过滤掉非数值答案）
    # pd.to_numeric尝试转换为数字，失败则设为NaN；notnull()保留有效值
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
    dataset = dataset.iloc[np.where(is_number)[0]]

    # 应用格式化函数到每一行，创建"Messages"列
    dataset["Messages"] = dataset.apply(format_row, axis=1)

    # 使用分词器的聊天模板将消息列表转换为模型可理解的文本字符串
    # apply_chat_template会自动添加<|im_start|>、<|im_end|>等特殊token
    dataset["text"] = tokenizer.apply_chat_template(
        dataset["Messages"].values.tolist(),  # 转换为列表
        tokenize=False  # 不进行tokenization，只生成文本
    )

    # 返回HuggingFace Dataset对象（从Pandas转换回来）
    return Dataset.from_pandas(dataset)


# ============================================================================
# 函数：监督微调（SFT）训练
# ============================================================================
def train_sft(model, tokenizer, dataset, swanlab_callback=None):
    """
    使用监督微调（Supervised Fine-Tuning）训练模型

    SFT是最基础的微调方法：在人工标注的（输入，输出）对上训练模型
    模型学习"当看到这个输入时，应该给出这样的输出"

    参数:
        model: PEFT模型
        tokenizer: 分词器
        dataset: 训练数据集（必须包含"text"字段）
        swanlab_callback: SwanLab实验追踪回调对象（可选）
    """
    # 创建SFT训练器
    trainer = SFTTrainer(
        model=model,  # 待训练的模型
        tokenizer=tokenizer,  # 分词器
        train_dataset=dataset,  # 训练数据集
        args=SFTConfig(
            dataset_text_field="text",  # 数据集中哪个字段包含训练文本
            per_device_train_batch_size=1,  # 每个GPU的批次大小（小批次节省显存）
            gradient_accumulation_steps=1,  # 梯度累积步数（1表示不累积）
            warmup_steps=5,  # 预热步数（学习率从0线性增加到预设值）
            num_train_epochs=1,  # 训练轮数（1轮通常足够LoRA微调）
            learning_rate=2e-4,  # 学习率（LoRA通常使用2e-4到5e-5）
            logging_steps=5,  # 每5步记录一次日志
            optim="adamw_8bit",  # 优化器（8位AdamW节省显存）
            weight_decay=0.01,  # 权重衰减（正则化，防止过拟合）
            lr_scheduler_type="linear",  # 学习率调度器类型（线性衰减）
            seed=SEED,  # 随机种子
            report_to="swanlab" if swanlab_callback else None,  # 实验追踪工具
        ),
        callbacks=[swanlab_callback] if swanlab_callback else None  # 回调函数列表
    )

    # 开始训练
    trainer.train()


# ============================================================================
# 奖励函数1：检查输出格式是否正确
# ============================================================================
def match_format_approximately(completions, **kwargs):
    """
    GRPO奖励函数：根据模型输出是否包含正确的推理和答案标记，给出分数

    参数:
        completions: 模型生成的完成结果列表，每个元素包含[{"content": "生成文本"}]
        **kwargs: 其他参数（未使用）

    返回:
        list: 每个生成结果的分数列表（浮点数）
    """
    scores = []
    for completion in completions:
        # 获取生成的文本内容
        response = completion[0]["content"]
        score = 0

        # 检查结束标记的出现次数：期望正好出现1次，否则扣分
        # REASONING_END标记：推理结束
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        # SOLUTION_START标记：答案开始
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        # SOLUTION_END标记：答案结束
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0

        scores.append(score)
    return scores


# ============================================================================
# 奖励函数2：检查答案是否正确
# ============================================================================
def check_answer(prompts, completions, answer, **kwargs):
    """
    GRPO奖励函数：根据模型生成的答案是否正确进行评分

    参数:
        prompts: 原始提示列表（未直接使用）
        completions: 模型生成的完成结果列表
        answer: 正确答案列表（从数据集中获取）
        **kwargs: 其他参数

    返回:
        list: 每个生成结果的分数列表
    """
    # 编译正则表达式，用于提取SOLUTION_START和SOLUTION_END之间的内容
    # re.DOTALL: 让.匹配换行符
    # re.MULTILINE: 多行模式
    match_format = re.compile(
        rf"{REASONING_END}.*?{SOLUTION_START}(.+?){SOLUTION_END}.*$",
        flags=re.MULTILINE | re.DOTALL
    )

    # 提取所有生成的内容
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for response, true_answer in zip(responses, answer):
        # 使用正则表达式搜索答案
        guess = match_format.search(response)

        if guess is None:
            # 没有找到答案格式：严重惩罚
            scores.append(-2.0)
        else:
            # 提取答案内容，与正确答案比较
            extracted_answer = guess.group(1).strip()
            if extracted_answer == true_answer.strip():
                # 答案正确：高分奖励
                scores.append(5.0)
            else:
                # 答案错误：中等惩罚
                scores.append(-1.5)

    return scores


# ============================================================================
# 函数：GRPO（Group Relative Policy Optimization）训练
# ============================================================================
def train_grpo(model, tokenizer, dataset, max_prompt_length, max_completion_length, swanlab_callback=None):
    """
    使用GRPO进行强化学习训练

    GRPO是一种PPO的变体，专门用于语言模型的强化学习微调。
    不需要训练critic网络，通过组内相对比较来优化策略。

    参数:
        model: PEFT模型
        tokenizer: 分词器
        dataset: 训练数据集（必须包含"prompt"和"answer"字段）
        max_prompt_length: 最大提示长度（输入部分）
        max_completion_length: 最大生成长度（输出部分）
        swanlab_callback: SwanLab实验追踪回调对象（可选）
    """

    # 配置vLLM的采样参数（控制生成文本的随机性）
    sampling_params = SamplingParams(
        min_p=0.1,  # 最小概率阈值（低于此概率的token被过滤）
        top_p=1.0,  # 核采样（1.0表示不截断）
        top_k=-1,  # Top-K采样（-1表示禁用）
        seed=SEED,  # 随机种子
        stop=[tokenizer.eos_token],  # 停止标记（遇到EOS token停止）
        include_stop_str_in_output=True  # 在输出中包含停止标记
    )

    # GRPO训练配置
    training_args = GRPOConfig(
        vllm_sampling_params=sampling_params,  # vLLM采样参数
        temperature=1.0,  # 温度参数（1.0为标准，越高越随机）
        learning_rate=5e-6,  # 学习率（RL微调通常比SFT更低）
        weight_decay=0.01,  # 权重衰减
        warmup_ratio=0.1,  # 预热比例（前10%步数线性增加学习率）
        lr_scheduler_type="linear",  # 学习率调度器
        optim="adamw_8bit",  # 优化器
        logging_steps=1,  # 每步都记录日志
        per_device_train_batch_size=1,  # 每设备批次大小
        gradient_accumulation_steps=1,  # 梯度累积步数
        num_generations=2,  # 每个提示生成几个候选（组大小）
        max_prompt_length=max_prompt_length,  # 最大提示长度
        max_completion_length=max_completion_length,  # 最大生成长度
        max_steps=10,  # 最大训练步数（示例中只训练10步）
        save_steps=100,  # 每100步保存一次检查点
        report_to="swanlab" if swanlab_callback else None,  # 实验追踪
        output_dir="outputs",  # 输出目录
    )

    # 创建GRPO训练器
    trainer = GRPOTrainer(
        model=model,  # 待优化的策略模型
        processing_class=tokenizer,  # 分词器（处理类）
        reward_funcs=[match_format_approximately, check_answer],  # 奖励函数列表
        args=training_args,  # 训练参数
        train_dataset=dataset,  # 训练数据集
        callbacks=[swanlab_callback] if swanlab_callback else None  # 回调函数
    )

    # 开始GRPO训练
    trainer.train()


# ============================================================================
# 函数：推理示例
# ============================================================================
def inference_example(model, tokenizer, prompt):
    """
    对模型进行推理测试（生成文本）

    参数:
        model: 训练好的模型
        tokenizer: 分词器
        prompt: 输入提示文本
    """
    # 将提示文本转换为模型输入张量
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")  # 移动到GPU

    # 使用模型的快速生成方法
    output_tokens = model.fast_generate(
        **inputs,
        max_new_tokens=1024,  # 最多生成1024个新token
        temperature=1.0,  # 温度（控制随机性）
        top_k=50,  # Top-K采样（只从概率最高的50个token中选择）
        top_p=1.0,  # 核采样（1.0表示不截断）
    )

    # 解码生成的token为文本
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    print("Inference output:", output_text)


# ============================================================================
# 主函数：执行完整的训练流程
# ============================================================================
def main():
    """
    主执行流程：
    1. 初始化模型（如果有GRPO保存的LoRA则加载）
    2. 配置SwanLab实验追踪
    3. SFT阶段：在OpenMathReasoning数据集上监督微调
    4. GRPO阶段：在DAPO-Math数据集上强化学习微调
    5. 保存最终的LoRA权重
    """

    # -------------------------
    # 步骤1: 初始化模型
    # -------------------------
    # 尝试加载之前GRPO训练保存的LoRA（如果存在），否则创建新LoRA
    # for_training=True 设置为训练模式
    model, tokenizer = init_model("grpo_saved_lora")

    # -------------------------
    # 步骤2: 配置 SwanLab 回调（实验追踪）
    # -------------------------
    # 设置SwanLab API密钥（用于上传实验数据）
    os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj"

    # 创建SwanLab回调对象，记录训练过程
    swanlab_callback = SwanLabCallback(
        project="Qwen3-VL-finetune",  # 项目名称
        experiment_name="xingkong_2000",  # 实验名称
        config={  # 实验配置（自动记录）
            "model": "Qwen/Qwen3-8B",  # 基础模型
            "dataset": "linxy/LaTeX_OCR",  # 数据集（注释可能与实际不符）
            "prompt": "不知道",  # 自定义提示
            "train_data_number": 2000,  # 训练数据数量
            "lora_rank": LORA_RANK,  # LoRA秩
            "lora_alpha": LORA_RANK * 2,  # LoRA alpha参数
            "lora_dropout": 0.0,  # LoRA dropout率
        },
    )

    # -------------------------
    # 步骤3: SFT（监督微调）阶段
    # -------------------------
    # 加载OpenMathReasoning数据集（思维链数据）
    # split="cot" 表示使用思维链（Chain of Thought）分割
    sft_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")

    # 打乱数据集并只取前5条（示例用小数据集）
    sft_dataset = sft_dataset.shuffle(seed=SEED).select(range(5))

    # 格式化数据集为SFT所需的聊天格式
    sft_dataset = format_sft_dataset(sft_dataset, tokenizer)

    # 执行SFT训练
    train_sft(model, tokenizer, sft_dataset, swanlab_callback=swanlab_callback)

    # -------------------------
    # 步骤4: GRPO（强化学习）阶段
    # -------------------------
    # 清理显存（删除SFT数据集，释放内存）
    del sft_dataset
    torch.cuda.empty_cache()  # 清空PyTorch缓存
    gc.collect()  # 强制垃圾回收

    # 加载DAPO-Math数据集（英文分割）
    grpo_dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")

    # 只取前5条（示例用小数据集）
    grpo_dataset = grpo_dataset.shuffle(seed=SEED).select(range(5))

    # 重新格式化数据集为GRPO需要的格式
    # 每条数据包含"prompt"（对话消息）和"answer"（正确答案）
    grpo_dataset = grpo_dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示
            {"role": "user", "content": x["prompt"]}  # 用户问题
        ],
        "answer": x["solution"]  # 正确答案（用于奖励函数）
    })

    # 计算长度限制
    max_prompt_length = 256  # 提示部分最大长度
    # 生成长度 = 总长度 - 提示长度
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length

    # 执行GRPO训练
    train_grpo(model, tokenizer, grpo_dataset,
               max_prompt_length, max_completion_length,
               swanlab_callback=swanlab_callback)

    # -------------------------
    # 步骤5: 保存最终的LoRA权重
    # -------------------------
    model.save_pretrained("grpo_saved_lora")

    tokenizer.save_pretrained("grpo_saved_lora")

# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    main()