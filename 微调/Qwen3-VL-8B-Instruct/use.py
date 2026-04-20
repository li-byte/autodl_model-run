import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from peft import PeftModel

# -----------------------------
# 设备设置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# 模型路径/权重
# -----------------------------
base_model_path = "../../models/Qwen3-VL-8B-Instruct"
lora_weights_path = "./qwen3-lora-final"

# -----------------------------
# Processor 加载
# -----------------------------
processor = AutoProcessor.from_pretrained(base_model_path)

# -----------------------------
# Chat 消息推理函数（增强 LoRA 风格）
# -----------------------------
def generate_from_messages(
    messages,
    use_lora=True,
    max_new_tokens=1024,
    num_beams=1,
    temperature=0.6,
    top_p=0.9,
    lora_alpha_scale=3.0  # 放大 LoRA 权重，让微调数据优先
):
    # -----------------------------
    # 加载基础模型
    # -----------------------------
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto"
    )

    # -----------------------------
    # 加载 LoRA 并放大低秩权重
    # -----------------------------
    if use_lora:
        model = PeftModel.from_pretrained(model, lora_weights_path)
        # 放大 LoRA 输出权重
        for n, p in model.named_parameters():
            if "lora" in n:
                p.data = p.data * lora_alpha_scale

    model.eval()

    # -----------------------------
    # 处理消息
    # -----------------------------
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # -----------------------------
    # 生成输出
    # -----------------------------
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,   # 开启采样增加多样性
            pad_token_id=processor.tokenizer.pad_token_id
        )

    # -----------------------------
    # 去掉输入部分，只保留生成文本
    # -----------------------------
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # -----------------------------
    # 显存释放
    # -----------------------------
    del model
    torch.cuda.empty_cache()

    return output_text

# -----------------------------
# 示例调用
# -----------------------------
if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "./data/images/dog_running.jpg"},
                {"type": "text", "text": "请详细描述图片内容"}
            ]
        }
    ]

    print("\n=== LoRA 优化生成 ===")
    result_lora = generate_from_messages(messages, use_lora=True)
    print("生成结果（LoRA模型）:", result_lora)