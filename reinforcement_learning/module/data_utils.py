# data_utils.py
import re
import numpy as np
import pandas as pd
from datasets import Dataset
from config import SYSTEM_PROMPT, REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END

# -----------------------------
# 格式化SFT数据集
# -----------------------------
def format_sft_dataset(dataset, tokenizer):
    """
    将原始COT数据集格式化为SFT训练需要的聊天格式
    """

    def format_row(x):
        thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
        final_prompt = REASONING_START + thoughts + REASONING_END + SOLUTION_START + x["expected_answer"] + SOLUTION_END
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["problem"]},
            {"role": "assistant", "content": final_prompt},
        ]

    # 数据清洗：保留数值类型的答案
    dataset = dataset.to_pandas()
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
    dataset = dataset.iloc[np.where(is_number)[0]]

    # 格式化消息
    dataset["Messages"] = dataset.apply(format_row, axis=1)

    # 使用tokenizer生成模型输入格式
    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)
    return Dataset.from_pandas(dataset)