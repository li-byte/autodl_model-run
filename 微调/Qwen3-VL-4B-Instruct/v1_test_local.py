import os
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import importlib


# ---------------------------
# 配置
# ---------------------------
PROMPT_TEXT = "图片内容是"
BASE_MODEL_ID = "../../models/Qwen3-VL-4B-Instruct"
PEFT_DIR = "./Qwen3-VL-4B"
MERGE_LORA_IN_MEMORY = True
NUM_TEST_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
LOCAL_DATASET_DIR = "data"


# ---------------------------
# 加载本地数据集
# ---------------------------
import os
import json
from PIL import Image

def load_local_dataset_json(dataset_dir: str, num_samples: int = 5):
    """
    加载本地 JSON 格式数据集
    :param dataset_dir: 数据集根目录
    :param num_samples: 加载样本数量
    :return: list of dict {"image": PIL.Image, "text": str, "filename": str}
    """
    json_file = os.path.join(dataset_dir, "dataset.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"未找到 JSON 数据集文件: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data[:num_samples]:
        img_path = os.path.join(dataset_dir, item["image"])
        if not os.path.exists(img_path):
            print(f"警告: 图片文件不存在 {img_path}, 已跳过")
            continue
        image = Image.open(img_path).convert("RGB")
        samples.append({"image": image, "text": item.get("text", ""), "filename": os.path.basename(img_path)})
    return samples

# ---------------------------
# 加载基础模型
# ---------------------------
def load_backbone(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    # 动态导入模型类
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
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
# 加载 LoRA
# ---------------------------
def load_lora_model(peft_dir: str, base_model_id: str = BASE_MODEL_ID):
    base_model, _base_tok, _base_proc = load_backbone(base_model_id)
    peft_model = PeftModel.from_pretrained(base_model, peft_dir)
    model = peft_model
    if MERGE_LORA_IN_MEMORY:
        try:
            model = peft_model.merge_and_unload()
            print("LoRA 内存合并成功。")
        except Exception:
            print("LoRA 内存合并失败，使用未合并模型。")
            model = peft_model
    model.to(dtype=DTYPE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(peft_dir, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(peft_dir, use_fast=False, trust_remote_code=True)
    return model, tokenizer, processor


# ---------------------------
# 构建模型输入
# ---------------------------
def build_inputs(processor, image, prompt_text: str):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, do_resize=True)
    return inputs


# ---------------------------
# 推理生成
# ---------------------------
@torch.inference_mode()
def generate_answer(model, tokenizer, processor, image, max_new_tokens: int = 512):
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
    gen_kwargs = {"input_ids": input_ids, "pixel_values": pixel_values,
                  "max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True}
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if image_grid_thw is not None:
        gen_kwargs["image_grid_thw"] = image_grid_thw
    outputs = model.generate(**gen_kwargs)
    gen_seq = outputs[0].tolist()
    prompt_len = input_ids.shape[1]
    gen_ids = gen_seq[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# ---------------------------
# 主函数
# ---------------------------
def main():
    samples = load_local_dataset_json(LOCAL_DATASET_DIR, NUM_TEST_SAMPLES)
    print(f"加载 {len(samples)} 个样本用于测试。")

    base_model, base_tokenizer, base_processor = load_backbone(BASE_MODEL_ID)
    base_model.eval()
    try:
        lora_model, lora_tokenizer, lora_processor = load_lora_model(PEFT_DIR, BASE_MODEL_ID)
        lora_model.eval()
    except Exception as e:
        print(f"LoRA 加载失败: {e}")
        lora_model = None
        lora_tokenizer = base_tokenizer
        lora_processor = base_processor

    for idx, sample in enumerate(samples):
        image = sample["image"]
        gt = sample.get("text", "")
        print(f"\n[样本 {idx} - {sample['filename']}]")
        print(f"真实值 (GT): {gt}")
        base_pred = generate_answer(base_model, base_tokenizer, base_processor, image)
        print(f"基础模型预测: {base_pred}")
        if lora_model:
            lora_pred = generate_answer(lora_model, lora_tokenizer, lora_processor, image)
            print(f"LoRA 微调预测: {lora_pred}")
        else:
            print("LoRA 微调模型: <未加载>")


if __name__ == "__main__":
    main()