
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LENGTH = 2048

REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"

SYSTEM_PROMPT = f"""你将得到一个问题。
请仔细思考并给出你的推理过程。
将推理过程放在 {REASONING_START} 和 {REASONING_END} 之间。
然后，将最终答案放在 {SOLUTION_START} 和 {SOLUTION_END} 之间。"""


def load_model(model_path="./full_finetuned_model"):
    """
    加载全量微调模型

    为什么要用 AutoModelForCausalLM？
    - 全量微调更新了模型的所有参数，保存的是完整模型权重
    - AutoModelForCausalLM 直接加载完整模型，适合推理
    - 不需要 PEFT/LoRA 相关库，因为模型已经是完整形态
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def test_inference(model, tokenizer, problem):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # 提取 assistant 回答
    if "assistant" in output_text:
        model_output = output_text.split("assistant")[-1].strip()
    else:
        model_output = output_text

    print(f"输出: {model_output}")
    print("-" * 50)


def main():
    model, tokenizer = load_model()
    test_inference(model, tokenizer, "描述一下小明的个人背景")


if __name__ == "__main__":
    main()