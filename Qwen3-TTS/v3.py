import os

# 设置 OpenMP 线程数，限制 CPU 并行线程，避免资源争抢
os.environ["OMP_NUM_THREADS"] = "4"
import time
import torch
import soundfile as sf
import random
import numpy as np
from qwen_tts import Qwen3TTSModel


def build_batch_inputs(texts, instruct):
    """
    构建 batch 输入（一次性推理）

    参数：
        texts: 列表，每个元素包含中英文文本
        instruct: 风格控制（中英文各一个）

    返回：
        batch_text: 展平后的文本列表（中英交替）
        batch_lang: 对应语言标签
        batch_instruct: 对应每条文本的风格控制
    """
    batch_text = []
    batch_lang = []
    batch_instruct = []

    for item in texts:
        # 每个 item 拆成 中文 + 英文
        batch_text.extend([item["Chinese"], item["English"]])

        # 对应语言标签
        batch_lang.extend(["Chinese", "English"])

        # instruct 是一个长度为2的列表（中英文风格）
        batch_instruct.extend(instruct)

    return batch_text, batch_lang, batch_instruct




def post_process(
        wavs,
        sr,
        texts,
        gap_zh_en,
        gap_between,
        repeat
):

    silence_zh_en = np.zeros(int(sr * gap_zh_en))
    silence_between = np.zeros(int(sr * gap_between))

    segments = []
    idx = 0

    for i, item in enumerate(texts):

        wav_zh = wavs[idx]
        wav_en = wavs[idx + 1]
        idx += 2

        segment_parts = []

        for _ in range(repeat):
            segment_parts.append(wav_zh)
            segment_parts.append(silence_zh_en)
            segment_parts.append(wav_en)

        segment_audio = np.concatenate(segment_parts)
        segments.append(segment_audio)

    # ✅ 打乱词条级别顺序
    random.shuffle(segments)

    # ✅ 拼接结果
    all_audio = []

    # 开头静音
    all_audio.append(silence_between)

    for i, seg in enumerate(segments):
        all_audio.append(seg)

        if i < len(segments) - 1:
            all_audio.append(silence_between)

    return np.concatenate(all_audio)


if __name__ == '__main__':

    # ===================== 模型加载 =====================
    model = Qwen3TTSModel.from_pretrained(
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",  # 模型名称
        device_map="cuda:0",               # 使用 GPU 0
        dtype=torch.bfloat16,              # 使用 bfloat16 降低显存占用
    )

    # 输入文本（中英文对照）
    texts = [
        {"Chinese": "向量数据库", "English": "milvus"},
        {"Chinese": "智能体", "English": "Agent"},
        {"Chinese": "检索增强生成", "English": "RAG"},
        {"Chinese": "火山引擎", "English": "HiAgent"},
        {"Chinese": "技能", "English": "Skills"},
        # {"Chinese": "龙猫大模型", "English": "LongCat-Next"},
        {"Chinese": "AI公司", "English": "Anthropic"},
        {"Chinese": "算力云平台", "English": "autodl"},
        {"Chinese": "向量数据库", "English": "seekdb"},
    ]

    speaker = "Serena"  # 声音角色

    # instruct[0] -> 中文风格
    # instruct[1] -> 英文风格
    instruct = ["温柔,甜美,快速", "soft, gentle, sweet, warm"]

    # 中英文之间停顿（秒）
    gap_zh_en = 1.0

    # 不同词条之间停顿（秒）
    gap_between = 2.0

    # 每个词条重复次数
    repeat = 2

    # ===================== 构建 batch =====================
    batch_text, batch_lang, instruct = build_batch_inputs(texts, instruct)

    # GPU 同步，确保计时准确
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # ===================== 🚀 单次推理 =====================
    wavs, sr = model.generate_custom_voice(
        text=batch_text,        # 批量文本
        language=batch_lang,    # 语言标签
        speaker=speaker,        # 声音
        instruct=instruct,      # 风格控制
    )

    # ===================== 后处理 =====================
    final_audio = post_process(
        wavs,
        sr,
        texts,
        gap_zh_en,
        gap_between,
        repeat
    )

    # 再次同步 GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    print(f"⏱️ 推理耗时: {end_time - start_time:.3f} 秒")

    # ===================== 保存音频 =====================
    sf.write("output_single_infer.wav", final_audio, sr)