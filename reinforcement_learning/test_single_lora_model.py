
# test_model.py
import os
import re
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

MAX_SEQ_LENGTH = 2048
LORA_RANK = 4
GPU_MEMORY_UTILIZATION = 0.7

REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"

SYSTEM_PROMPT = f"""你将得到一个问题。
请仔细思考并给出你的推理过程。
将推理过程放在 {REASONING_START} 和 {REASONING_END} 之间。
然后，将最终答案放在 {SOLUTION_START} 和 {SOLUTION_END} 之间。"""


def load_model(model_path="../../models/Qwen3-4B", lora_path="programming_lora"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def test_inference(model, tokenizer, problem):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_k=50, top_p=0.9)

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # 只提取模型生成的回答部分（assistant 后面的内容）
    if "user" in output_text:
        parts = output_text.split("user")
        if len(parts) > 1:
            # 取最后一个 user 后面的内容
            user_content = parts[-1].strip()
            # 再分离出问题（如果有换行）
            lines = user_content.split('\n', 1)
            if len(lines) > 1:
                model_output = lines[1].strip()
            else:
                model_output = user_content
        else:
            model_output = output_text
    else:
        model_output = output_text

    # 只输出 user 后面的模型生成部分
    print(f"输出: {model_output}")
    print("-" * 50)


def main():
    model, tokenizer = load_model()

    test_cases = [
        "某某写什么代码",
    ]

    for problem in test_cases:
        test_inference(model, tokenizer, problem)


if __name__ == "__main__":
    main()
