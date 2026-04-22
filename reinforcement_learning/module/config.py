import os

# -----------------------------
# 环境变量
# -----------------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj"

# -----------------------------
# 模型参数
# -----------------------------
MAX_SEQ_LENGTH = 2048
LORA_RANK = 4
SEED = 3407
GPU_MEMORY_UTILIZATION = 0.7

# -----------------------------
# 系统提示和答案标记
# -----------------------------
REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"

SYSTEM_PROMPT = f"""你将得到一个问题。
请仔细思考并给出你的推理过程。
将推理过程放在 {REASONING_START} 和 {REASONING_END} 之间。
然后，将最终答案放在 {SOLUTION_START} 和 {SOLUTION_END} 之间。"""