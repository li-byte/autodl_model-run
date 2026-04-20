import os
import sys
from typing import List, Tuple
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import importlib

# 自定义视觉处理工具
from qwen_vl_utils import process_vision_info

# ---------------------------
# 全局配置
# ---------------------------
PROMPT_TEXT = "Transcribe the LaTeX of this image."

# 模型与 LoRA 路径
# BASE_MODEL_ID = "/root/autodl-fs/Qwen3-VL-30B-A3B-Instruct"
# PEFT_DIR = "/root/autodl-fs/output/Qwen3-VL-30B"
BASE_MODEL_ID = "../../models/Qwen3-VL-4B-Instruct"
PEFT_DIR = "./Qwen3-VL-4B"

# 是否在内存中合并 LoRA（不落盘）
MERGE_LORA_IN_MEMORY = True
# 推理样本数量
NUM_TEST_SAMPLES = 5

# 设备与数据类型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


# ---------------------------
# 加载基础模型（未微调）
# ---------------------------
def load_backbone(model_id: str):
    """
    加载基础模型、tokenizer 和 processor
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True
    )

    # 动态导入模型类
    config = AutoConfig.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME", "./"), trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    module_name = f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    module = importlib.import_module(module_name)
    model_cls = getattr(module, arch)

    # 实例化模型
    model = model_cls.from_pretrained(
        model_id,
        cache_dir=os.environ.get("HF_HOME", "./"),
        device_map="auto" if DEVICE.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.to(dtype=DTYPE)
    return model, tokenizer, processor


# ---------------------------
# 加载 LoRA 微调模型
# ---------------------------
def load_lora_model(peft_dir: str, base_model_id: str = BASE_MODEL_ID):
    """
    从 PEFT LoRA 目录加载微调模型
    """
    if not os.path.isdir(peft_dir):
        raise FileNotFoundError(f"未找到微调模型目录: {peft_dir}")

    # 加载基础模型
    base_model, _base_tok, _base_proc = load_backbone(base_model_id)

    # 加载 LoRA 微调权重
    peft_model = PeftModel.from_pretrained(base_model, peft_dir)
    model = peft_model
    if MERGE_LORA_IN_MEMORY:
        try:
            # 在内存中合并 LoRA，释放额外显存
            model = peft_model.merge_and_unload()
            print("LoRA内存合并成功。")
        except Exception:
            print("警告: LoRA内存合并失败，继续使用未合并模型。")
            model = peft_model
    model.to(dtype=DTYPE)
    model.eval()

    # tokenizer/processor 从 LoRA 目录读取，保证词表与模板一致
    tokenizer = AutoTokenizer.from_pretrained(peft_dir, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(peft_dir, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True)
    return model, tokenizer, processor


# ---------------------------
# 构建模型输入
# ---------------------------
def build_inputs(processor, image, prompt_text: str):
    """
    构建图像 + 文本的模型输入
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    # 使用 processor 将 messages 转为模型可接受格式
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, do_resize=True)
    return inputs


# ---------------------------
# 确保 LaTeX 使用双美元符号块
# ---------------------------
def ensure_block_dollars(text: str) -> str:
    """
    将单美元符号或空字符串转为 $$...$$ 块
    """
    if text is None:
        return "$$$$"
    s = str(text).strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s
    if s.startswith("$") and s.endswith("$") and not s.startswith("$$") and not s.endswith("$$"):
        inner = s[1:-1].strip()
        return f"$${inner}$$"
    if s.count("$$") >= 2:
        return s
    return f"$${s}$$"


# ---------------------------
# 推理生成函数
# ---------------------------
@torch.inference_mode()
def generate_answer(model, tokenizer, processor, image, max_new_tokens: int = 512) -> str:
    """
    给定图像，生成 LaTeX 文本
    """
    inputs = build_inputs(processor, image, PROMPT_TEXT)

    input_ids = torch.as_tensor(inputs["input_ids"], device=DEVICE)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = torch.as_tensor(attention_mask, device=DEVICE)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

    pixel_values = torch.as_tensor(inputs.get("pixel_values"), device=DEVICE)
    image_grid_thw = torch.as_tensor(inputs.get("image_grid_thw"), device=DEVICE)

    # 生成参数
    gen_kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if image_grid_thw is not None:
        gen_kwargs["image_grid_thw"] = image_grid_thw

    outputs = model.generate(**gen_kwargs)
    gen_seq = outputs[0].tolist()
    prompt_len = input_ids.shape[1]
    gen_ids = gen_seq[prompt_len:]  # 只保留生成部分
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# ---------------------------
# 主函数
# ---------------------------
def main():
    print("Loading dataset linxy/LaTeX_OCR (synthetic_handwrite)...")
    ds = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")
    ds = ds.shuffle(seed=222)
    test_split = ds["test"].select(range(NUM_TEST_SAMPLES))

    print("Loading base model...")
    base_model, base_tokenizer, base_processor = load_backbone(BASE_MODEL_ID)
    base_model.eval()

    print(f"Loading LoRA fine-tuned model from: {PEFT_DIR}")
    try:
        lora_model, lora_tokenizer, lora_processor = load_lora_model(PEFT_DIR, BASE_MODEL_ID)
        lora_model.eval()
    except Exception as e:
        print(f"加载微调模型失败: {e}")
        print("仅对基础模型进行推理对比。")
        lora_model = None
        lora_tokenizer = base_tokenizer
        lora_processor = base_processor

    # 依次对样本进行推理对比
    print(f"\n===== Inference Comparison on {NUM_TEST_SAMPLES} samples =====\n")
    for idx, sample in enumerate(test_split):
        image = sample["image"]
        gt = sample.get("text", "")
        print(f"[Sample {idx}]------------------------------")
        print(f"GT: {ensure_block_dollars(gt)}")

        base_pred = ensure_block_dollars(generate_answer(base_model, base_tokenizer, base_processor, image))
        print(f"Base: {base_pred}")

        if lora_model is not None:
            lora_pred = ensure_block_dollars(generate_answer(lora_model, lora_tokenizer, lora_processor, image))
            print(f"LoRA: {lora_pred}")
        else:
            print("LoRA: <not loaded>")

        print()


if __name__ == "__main__":
    main()