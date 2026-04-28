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

# LoRA配置
LORA_PATHS = {
    "family": "family_relation_lora",
    "programming": "programming_lora",
}

CLASSIFY_PROMPT = """请判断以下问题属于哪个类别，只返回类别名称family或者programming或者other 不需要额外输出。
类别：family（家庭关系）、programming（编程）,other (其他)

问题：{question}
类别："""

def load_lora_for_inference(base_model, lora_name):
    """动态加载 LoRA，只返回带 LoRA 的模型"""
    lora_path = LORA_PATHS.get(lora_name)
    if lora_path and os.path.exists(lora_path):
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        return lora_model
    return base_model

def unload_lora(lora_model):
    """卸载 LoRA 并释放显存"""
    if isinstance(lora_model, PeftModel):
        base_model = lora_model.base_model
        del lora_model
        torch.cuda.empty_cache()
        return base_model
    return lora_model


def extract_category_from_output(output_text):
    """从模型输出中提取类别

    示例输入：
    "请判断以下问题属于哪个类别，只返回类别名称family或者programming 不需要额外输出。\n类别：family（家庭关系）、programming（编程）\n\n问题：晓斌喜欢写什么代码？\n类别：programming"

    返回："programming"
    """
    # 方法1：查找最后一个"类别："后面的内容
    if "类别：" in output_text:
        # 获取最后一个"类别："之后的部分
        last_category = output_text.rstrip().split("类别：")[-1]
        # 提取纯类别名（去除空格和可能的标点）
        category = last_category.strip().split()[0] if last_category.strip() else ""
        # 清理可能的标点符号
        category = category.strip('，。、\n\r\t')
        return category

    return "general"

def load_models(base_model_path="../../models/Qwen3-4B"):
    # 加载基础模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    # 预加载所有LoRA
    lora_models = {}
    for name, path in LORA_PATHS.items():
        if os.path.exists(path):
            lora_models[name] = PeftModel.from_pretrained(model, path)
            print(f"已加载LoRA: {name}")

    return model, tokenizer, lora_models


def classify_question(base_model, tokenizer, question):
    """使用基础模型判断问题类别"""
    prompt = CLASSIFY_PROMPT.format(question=question)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = base_model.generate(**inputs, max_new_tokens=10, temperature=0.1)

    category = extract_category_from_output(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # 提取类别
    for cat in ["family", "programming", "general"]:
        if cat in category.lower():
            return cat
    return "general"


def test_inference(base_model, tokenizer, problem):
    # 分类选择 LoRA
    selected_lora_name = classify_question(base_model, tokenizer, problem)
    print(f"[选择的LoRA类别: {selected_lora_name}]")

    # 动态加载 LoRA
    active_model = load_lora_for_inference(base_model, selected_lora_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_tokens = active_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # 清理 assistant 前缀
    if "assistant" in output_text:
        parts = output_text.split("assistant")
        if len(parts) > 1:
            output_text = parts[-1].strip()

    # 提取推理和答案
    reasoning_match = re.search(rf"{REASONING_START}(.+?){REASONING_END}", output_text, re.DOTALL)
    answer_match = re.search(rf"{SOLUTION_START}(.+?){SOLUTION_END}", output_text, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer = answer_match.group(1).strip() if answer_match else None

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

    # 卸载 LoRA，保持基础模型
    active_model = unload_lora(active_model)

def main():
    base_model, tokenizer, _ = load_models()  # 只加载基础模型

    test_cases = [
        "陈嘉宇的叔叔是？",
        "陈嘉宇喜欢写什么代码？",
        "陈嘉宇不喜欢写什么代码？",
        "陈嘉宇的职业是什么？",
        "陈嘉宇不喜欢Python，那么他最喜欢推荐的编程语言是什么？",
        "天空是蔚蓝色，窗外有什么",
        "陈嘉宇打算干嘛",
        "小蓝在干什么",
    ]

    for problem in test_cases:
        test_inference(base_model, tokenizer, problem)

if __name__ == "__main__":
    main()