import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import importlib
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
# 自定义视觉处理工具，用于处理图像和视频输入
from qwen_vl_utils import process_vision_info

# ---------------------------
# 全局配置
# ---------------------------

# 模型推理时的文本提示，用于指导模型将图像转录为 LaTeX
PROMPT_TEXT = "转录此图像的LaTeX。"

# 模型路径配置
BASE_MODEL_ID = "../../models/Qwen3-VL-4B-Instruct"  # 基础模型目录
PEFT_DIR = "./Qwen3-VL-4B"  # LoRA 微调权重目录

# 是否在内存中合并 LoRA 权重，避免生成新的模型文件，节省显存
MERGE_LORA_IN_MEMORY = True

# 推理样本数量
NUM_TEST_SAMPLES = 5

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据类型配置：CUDA 使用 bfloat16 加速，CPU 使用 float32
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


# ---------------------------
# 加载基础模型（未微调）
# ---------------------------
def load_backbone(model_id: str):
    """
    加载基础模型、tokenizer 和 processor
    :param model_id: 基础模型目录或 HuggingFace 模型 ID
    :return: model, tokenizer, processor
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=os.environ.get("HF_HOME", "./"),  # 指定缓存目录
        use_fast=False,
        trust_remote_code=True
    )

    # 加载多模态处理器（处理文本、图像、视频）
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=os.environ.get("HF_HOME", "./"),
        use_fast=False,
        trust_remote_code=True
    )

    # 加载基础视觉语言模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto" if DEVICE.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.to(dtype=DTYPE)  # 设置模型数据类型
    return model, tokenizer, processor


# ---------------------------
# 加载 LoRA 微调模型
# ---------------------------
def load_lora_model(peft_dir: str, base_model_id: str = BASE_MODEL_ID):
    """
    从 PEFT LoRA 目录加载微调模型
    :param peft_dir: LoRA 权重目录
    :param base_model_id: 基础模型路径
    :return: model, tokenizer, processor
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
            # 在内存中合并 LoRA，减少显存占用
            model = peft_model.merge_and_unload()
            print("LoRA内存合并成功。")
        except Exception:
            print("警告: LoRA内存合并失败，继续使用未合并模型。")
            model = peft_model

    model.to(dtype=DTYPE)
    model.eval()  # 设置为推理模式

    # tokenizer/processor 从 LoRA 目录读取，保证词表和模板与微调一致
    tokenizer = AutoTokenizer.from_pretrained(
        peft_dir, use_fast=False, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        peft_dir, use_fast=False, trust_remote_code=True
    )

    return model, tokenizer, processor


# ---------------------------
# 构建模型输入
# ---------------------------
def build_inputs(processor, image, prompt_text: str):
    """
    构建图像 + 文本的模型输入
    :param processor: AutoProcessor 对象
    :param image: PIL.Image 或 ndarray
    :param prompt_text: 提示文本
    :return: 模型输入字典
    """
    # 定义消息列表，包含图像和文本
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # 图像输入
                {"type": "text", "text": prompt_text},  # 文本提示
            ],
        }
    ]

    # 将 messages 转为模型可接受格式
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)  # 处理视觉输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True
    )
    return inputs


# ---------------------------
# 推理生成函数
# ---------------------------
@torch.inference_mode()
def generate_answer(model, tokenizer, processor, image, max_new_tokens: int = 512) -> str:
    """
    给定图像，生成 LaTeX 文本
    :param model: 预训练或微调模型
    :param tokenizer: 分词器
    :param processor: 多模态处理器
    :param image: 输入图像
    :param max_new_tokens: 最大生成 token 数量
    :return: 生成的 LaTeX 文本
    """
    inputs = build_inputs(processor, image, PROMPT_TEXT)

    # 将输入转为 tensor
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
        "do_sample": False,  # 禁用采样，使用贪婪解码
        "use_cache": True,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if image_grid_thw is not None:
        gen_kwargs["image_grid_thw"] = image_grid_thw

    # 模型生成输出
    outputs = model.generate(**gen_kwargs)

    # 获取生成序列
    gen_seq = outputs[0].tolist()
    prompt_len = input_ids.shape[1]
    gen_ids = gen_seq[prompt_len:]  # 去掉输入 prompt 部分
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# ---------------------------
# 主函数
# ---------------------------
def main():
    # 加载数据集
    print("正在加载数据集 linxy/LaTeX_OCR (synthetic_handwrite)...")
    ds = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")
    ds = ds.shuffle(seed=222)
    test_split = ds["test"].select(range(NUM_TEST_SAMPLES))  # 只取前 N 个样本

    # 加载基础模型
    print("正在加载基础模型...")
    base_model, base_tokenizer, base_processor = load_backbone(BASE_MODEL_ID)
    base_model.eval()

    # 加载 LoRA 微调模型
    print(f"正在加载 LoRA 微调模型，路径: {PEFT_DIR}")
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
    print(f"\n===== 对 {NUM_TEST_SAMPLES} 个样本进行推理对比 =====\n")
    for idx, sample in enumerate(test_split):
        image = sample["image"]
        gt = sample.get("text", "")
        print(f"[样本 {idx}]------------------------------")
        print(f"真实值 (GT): {gt}")

        base_pred = generate_answer(base_model, base_tokenizer, base_processor, image)
        print(f"基础模型预测: {base_pred}")

        if lora_model is not None:
            lora_pred = generate_answer(lora_model, lora_tokenizer, lora_processor, image)
            print(f"LoRA 微调模型预测: {lora_pred}")
        else:
            print("LoRA 微调模型: <未加载>")

        print()


if __name__ == "__main__":
    main()