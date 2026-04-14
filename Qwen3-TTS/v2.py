import os
os.environ["OMP_NUM_THREADS"] = "4"

import time
import torch
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel


def build_batch_inputs(texts):
    """
    构建 batch 输入（一次性推理）
    """
    batch_text = []
    batch_lang = []

    for item in texts:
        batch_text.extend([item["Chinese"], item["English"]])
        batch_lang.extend(["Chinese", "English"])

    return batch_text, batch_lang


def post_process(
    wavs,
    sr,
    texts,
    gap_zh_en,
    gap_between,
    repeat
):
    """
    按原 process_item 逻辑拼接音频
    """

    silence_zh_en = np.zeros(int(sr * gap_zh_en))
    silence_between = np.zeros(int(sr * gap_between))

    all_audio = []

    idx = 0  # wavs 指针

    # 开头加静音
    all_audio.append(silence_between)

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
        all_audio.append(segment_audio)

        # 条目之间间隔
        if i < len(texts) - 1:
            all_audio.append(silence_between)

    return np.concatenate(all_audio)


if __name__ == '__main__':

    # ===================== 模型加载 =====================
    model = Qwen3TTSModel.from_pretrained(
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    texts = [
        {"Chinese": "向量数据库", "English": "milvus"},
        {"Chinese": "智能体", "English": "Agent"},
        {"Chinese": "检索增强生成", "English": "RAG"},
        {"Chinese": "火山引擎", "English": "HiAgent"},
        {"Chinese": "技能", "English": "skill"},
    ]

    speaker = "Serena"
    instruct = ""

    gap_zh_en = 1.0
    gap_between = 2.0
    repeat = 2

    # ===================== 构建 batch =====================
    batch_text, batch_lang = build_batch_inputs(texts)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # ===================== 🚀 单次推理 =====================
    wavs, sr = model.generate_custom_voice(
        text=batch_text,
        language=batch_lang,
        speaker=speaker,
        instruct=instruct,
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    print(f"⏱️ 推理耗时: {end_time - start_time:.3f} 秒")

    # ===================== 保存 =====================
    sf.write("output_single_infer.wav", final_audio, sr)