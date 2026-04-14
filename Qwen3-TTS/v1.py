import os
os.environ["OMP_NUM_THREADS"] = "4"

import time
import torch
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel


def generate_segment_cached(model, cache, text, language, speaker, instruct):
    """
    带缓存的生成函数（避免重复推理）
    """
    key = (text, language, speaker, instruct)

    if key not in cache:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
        cache[key] = (wavs[0], sr)

    return cache[key]


if __name__ == '__main__':

    # ===================== 模型加载 =====================
    model = Qwen3TTSModel.from_pretrained(
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    # ===================== 文本数据 =====================
    texts = [
        {"Chinese": "向量数据库", "English": "milvus"},
        {"Chinese": "大模型", "English": "large language model"},
    ]

    speaker = "Serena"
    instruct = "温柔,朗读"

    # ===================== 参数控制 =====================
    gap_zh_en = 1.0    # 中文 -> 英文间隔（秒）
    gap_between = 2.0  # 条目之间间隔（秒）
    repeat = 2         # 每条重复次数

    # ===================== 缓存 & 音频容器 =====================
    cache = {}
    all_audio = []
    sr = None

    # ===================== CUDA 同步 =====================
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # ===================== 主逻辑 =====================
    for idx, item in enumerate(texts):

        # ✅ 只生成一次（带缓存）
        wav_zh, sr = generate_segment_cached(
            model, cache,
            item["Chinese"], "Chinese", speaker, instruct
        )

        wav_en, sr = generate_segment_cached(
            model, cache,
            item["English"], "English", speaker, instruct
        )

        # 静音
        silence_zh_en = np.zeros(int(sr * gap_zh_en))

        # 🔁 重复拼接（不重复推理）
        for _ in range(repeat):
            all_audio.append(wav_zh)
            all_audio.append(silence_zh_en)
            all_audio.append(wav_en)

        # 条目间间隔
        if idx < len(texts) - 1:
            silence_between = np.zeros(int(sr * gap_between))
            all_audio.append(silence_between)

    # ===================== 拼接 =====================
    final_audio = np.concatenate(all_audio)

    # ===================== CUDA 同步 =====================
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    print(f"⏱️ 推理耗时: {end_time - start_time:.3f} 秒")

    # ===================== 保存 =====================
    sf.write(
        "output_custom_voice.wav",
        final_audio,
        sr
    )