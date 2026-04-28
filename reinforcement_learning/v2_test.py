
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


def load_model(model_path="../../models/Qwen3-4B", lora_path="family_relation_lora"):
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

    # 提取用户输入之后的模型输出部分
    if "assistant" in output_text:
        # 根据chat template提取assistant的回复
        parts = output_text.split("assistant")
        if len(parts) > 1:
            output_text = parts[-1].strip()

    # 尝试提取标签内容
    reasoning_match = re.search(rf"{REASONING_START}(.+?){REASONING_END}", output_text, re.DOTALL)
    answer_match = re.search(rf"{SOLUTION_START}(.+?){SOLUTION_END}", output_text, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer = answer_match.group(1).strip() if answer_match else None

    # 如果标签提取失败，显示完整输出以便调试
    if not reasoning or not answer:
        print(f"\n问题: {problem}")
        print(f"完整输出: {output_text}")
        print(f"推理: {reasoning if reasoning else '未找到标签'}")
        print(f"答案: {answer if answer else '未找到标签'}")
    else:
        print(f"\n问题: {problem}")
        print(f"推理: {reasoning}")
        print(f"答案: {answer}")
    print("-" * 50)


def main():
    model, tokenizer = load_model()

    test_cases = [
        "晓斌喜欢写什么代码？",
        "晓斌不喜欢写什么代码？",
        "晓斌的职业是什么？",
        "如果晓斌是Java架构师，他喜欢写什么代码？",
        "晓斌不喜欢Python，那么他最喜欢推荐的编程语言是什么？",
        "小明的爸爸是？",
        "天空是蔚蓝色，窗外有什么",
    ]

    for problem in test_cases:
        test_inference(model, tokenizer, problem)


if __name__ == "__main__":
    main()
