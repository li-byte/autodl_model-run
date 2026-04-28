# config.py
import os


# -----------------------------
# 配置参数
# -----------------------------
MAX_SEQ_LENGTH = 2048  # 模型输入的最大序列长度
LORA_RANK = 4  # LoRA的秩，用于低秩适配
SEED = 3407  # 随机种子，保证实验可复现
GPU_MEMORY_UTILIZATION = 0.7  # GPU显存占用比例限制

# -----------------------------
# 系统提示和答案标记
# -----------------------------
REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"

# 中文系统提示模板，指导模型输出推理过程和答案
SYSTEM_PROMPT = f"""你将得到一个问题。
请仔细思考并给出你的推理过程。
将推理过程放在 {REASONING_START} 和 {REASONING_END} 之间。
然后，将最终答案放在 {SOLUTION_START} 和 {SOLUTION_END} 之间。"""