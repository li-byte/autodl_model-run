import pandas as pd
import numpy as np
from datasets import Dataset
from config import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END, SYSTEM_PROMPT


def format_sft_dataset(dataset, tokenizer):
    """
    将原始COT数据集格式化为SFT训练所需的聊天格式。

    参数:
        dataset (Dataset): 原始COT数据集，每条样本包含 'problem', 'generated_solution', 'expected_answer'。
        tokenizer: 模型对应的分词器，用于生成模型可接受的输入文本格式。

    返回:
        Dataset: 已格式化的SFT训练数据集，包含 'text' 字段，可直接用于SFTTrainer。
    """

    # -----------------------------
    # 1. 定义行级格式化函数
    # -----------------------------
    def format_row(x):
        """
        对每一行样本生成聊天消息结构
        - thoughts: 模型推理过程，去掉 <think> 标签
        - final_prompt: 拼接推理过程和最终答案，并加上标记
        返回三条消息列表（system, user, assistant）
        """
        thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
        final_prompt = REASONING_START + thoughts + REASONING_END + SOLUTION_START + x["expected_answer"] + SOLUTION_END
        return [
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统消息，提供任务和输出格式指导
            {"role": "user", "content": x["problem"]},  # 用户消息，问题文本
            {"role": "assistant", "content": final_prompt},  # 模型回复，包含推理过程和答案标记
        ]

    # -----------------------------
    # 2. 数据清洗
    # -----------------------------
    # 将 HuggingFace Dataset 转为 pandas DataFrame
    dataset = dataset.to_pandas()

    # 保留数值类型的答案，避免非数值导致训练异常
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
    dataset = dataset.iloc[np.where(is_number)[0]]

    # -----------------------------
    # 3. 格式化消息
    # -----------------------------
    dataset["Messages"] = dataset.apply(format_row, axis=1)

    # -----------------------------
    # 4. 生成模型输入文本
    # -----------------------------
    # 使用 tokenizer.apply_chat_template 将消息列表转换为模型输入文本
    # 参数 tokenize=False: 不进行编码，只生成文本
    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)

    # 转回 HuggingFace Dataset
    return Dataset.from_pandas(dataset)


def format_grpo_dataset(dataset, tokenizer):
    """
    将原始GRPO数据集格式化为强化学习训练所需格式。

    参数:
        dataset (Dataset): 原始GRPO数据集，每条样本包含 'prompt', 'solution'。

    返回:
        Dataset: 已格式化GRPO数据集，包含:
            - prompt: 聊天消息列表（system + user）
            - answer: 正确答案，用于奖励函数计算
    """
    from config import SYSTEM_PROMPT

    return dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统消息，提供任务和输出规范
            {"role": "user", "content": x["prompt"]}  # 用户消息，问题文本
        ],
        "answer": x["solution"]  # 正确答案，用于奖励函数
    })