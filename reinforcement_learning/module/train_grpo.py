from trl import GRPOTrainer, GRPOConfig
from vllm import SamplingParams
from config import SEED, MAX_SEQ_LENGTH


def match_format_approximately(completions):
    """
    奖励函数1：检查生成文本中是否包含推理和答案标记。

    参数:
        completions (list): 模型生成的多轮回复，每个completion是 [{"role": "assistant", "content": str}, ...]。

    返回:
        scores (list[float]): 每条completion的奖励分数。

    逻辑：
        - 包含 <结束推理> +0.5，否则 -1.0
        - 包含 <答案> +0.5，否则 -1.0
        - 包含 <答案结束> +0.5，否则 -1.0
    """
    REASONING_END = "<结束推理>"
    SOLUTION_START = "<答案>"
    SOLUTION_END = "<答案结束>"

    scores = []
    for completion in completions:
        response = completion[0]["content"]  # 取模型生成文本
        score = 0
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(completions, answer):
    """
    奖励函数2：根据模型生成答案是否正确评分。

    参数:
        completions (list): 模型生成的回复
        answer (list): 真实答案列表

    返回:
        scores (list[float]): 每条completion对应的奖励分
        - 正确答案: +5.0
        - 错误答案: -1.5
        - 未生成答案格式: -2.0
    """
    import re
    REASONING_END = "<结束推理>"
    SOLUTION_START = "<答案>"
    SOLUTION_END = "<答案结束>"

    # 正则匹配推理结束后到答案结束标记之间的内容
    match_format = re.compile(
        rf"{REASONING_END}.*?{SOLUTION_START}(.+?){SOLUTION_END}.*$", flags=re.MULTILINE | re.DOTALL
    )

    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for r, true_answer in zip(responses, answer):
        guess = match_format.search(r)
        if guess is None:
            scores.append(-2.0)  # 没有正确格式
        else:
            # 生成答案和真实答案一致给 +5，否则 -1.5
            scores.append(5.0 if guess.group(1).strip() == true_answer.strip() else -1.5)
    return scores


def train_grpo(model, tokenizer, dataset, max_prompt_length, max_completion_length, swanlab_callback=None):
    """
    使用 GRPO（强化学习）对模型进行微调训练。

    参数:
        model: 待训练模型（包含LoRA）
        tokenizer: 分词器，用于处理prompt和生成文本
        dataset: 已格式化GRPO数据集，包含 'prompt' 和 'answer'
        max_prompt_length (int): 模型输入最大长度
        max_completion_length (int): 模型输出生成最大长度
        swanlab_callback: SwanLab实验追踪回调，可选

    步骤:
        1. 配置 vLLM 采样参数
        2. 初始化 GRPOTrainer
        3. 传入奖励函数 match_format_approximately & check_answer
        4. 启动训练
    """
    # -----------------------------
    # 1. vLLM采样参数
    # -----------------------------
    sampling_params = SamplingParams(
        min_p=0.1,  # 生成概率下限，避免极低概率token
        top_p=1.0,  # top-p采样阈值
        top_k=-1,  # top-k采样（-1表示禁用）
        seed=SEED,  # 随机种子
        stop=[tokenizer.eos_token],  # 停止生成标记
        include_stop_str_in_output=True
    )

    # -----------------------------
    # 2. 初始化GRPO训练器
    # -----------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[match_format_approximately, check_answer],  # 奖励函数
        args=GRPOConfig(
            vllm_sampling_params=sampling_params,
            temperature=1.0,  # 生成温度
            learning_rate=5e-6,  # 学习率
            weight_decay=0.01,  # 权重衰减
            warmup_ratio=0.1,  # 学习率warmup比例
            lr_scheduler_type="linear",
            optim="adamw_8bit",  # 8bit优化器，节省显存
            logging_steps=1,  # 每1步打印日志
            per_device_train_batch_size=1,  # 每个GPU batch size
            gradient_accumulation_steps=1,
            num_generations=2,  # 每条prompt生成数量
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=20,  # 总训练步数
            save_steps=100,  # 保存模型频率
            report_to="swanlab" if swanlab_callback else None,
            output_dir="outputs",
        ),
        train_dataset=dataset,
        callbacks=[swanlab_callback] if swanlab_callback else None
    )

    # -----------------------------
    # 3. 启动训练
    # -----------------------------
    trainer.train()