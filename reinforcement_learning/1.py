# test_lora_inference.py
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
# -----------------------------
# 配置参数
# -----------------------------
MODEL_PATH = "../../models/Qwen3-4B"
LORA_PATH = "grpo_saved_lora"
PROMPT = "101的平方分数是多少？"
MAX_NEW_TOKENS = 128

# -----------------------------
# 推理函数
# -----------------------------
def generate_text(model, tokenizer, prompt):
    """
    使用 HuggingFace 风格生成文本
    """
    model.eval()
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    output_tokens = model.fast_generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return output_text

# -----------------------------
# 主函数
# -----------------------------
def main():
    # 1. 基础模型
    print("=== 基础模型推理 ===")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=False,  # 注意：这里不启用 fast_inference，避免 FX 报错
    )
    base_output = generate_text(base_model, tokenizer, PROMPT)
    print(base_output)

    # 2. 强化后模型（加载 LoRA）
    print("\n=== 强化后模型推理（LoRA 加载） ===")
    finetuned_model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=False,
    )
    finetuned_model = PeftModel.from_pretrained(base_model, "grpo_saved_lora")
    finetuned_output = generate_text(finetuned_model, tokenizer, PROMPT)

    print(finetuned_output)

if __name__ == "__main__":
    main()