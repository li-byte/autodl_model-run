import os

# 设置 OpenMP 线程数（控制 CPU 并行线程数，避免线程过多导致资源争抢）
os.environ["OMP_NUM_THREADS"] = "4"

import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

if __name__ == '__main__':

    # ===================== 模型加载 =====================
    model = Qwen3TTSModel.from_pretrained(
        "../../models/Qwen3-TTS-12Hz-1.7B-CustomVoice",  # 本地模型路径
        device_map="cuda:0",      # 指定使用 GPU 0（如果是多卡可以改成 auto）
        dtype=torch.bfloat16,     # 使用 bfloat16 精度（节省显存，提高推理速度）
         attn_implementation="flash_attention_2",  # 使用 FlashAttention2 加速注意力计算/
    )

    # ===================== 单条语音生成 =====================

    # （新增）确保 CUDA 预热/同步，避免异步计时偏差
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # （新增）开始计时（模型加载完成之后）
    start_time = time.perf_counter()

    wavs, sr = model.generate_custom_voice(
        text="""秋天悄然降临时，天地仿佛被重新调色。山林间的枫树最先感知季节的变化，叶片由深绿渐渐转为橙黄、绯红，像被阳光长久浸染后的火焰，在枝头层层叠叠地铺展。微风拂过，林间响起轻微的沙沙声，一片片枫叶便在这温柔的召唤中缓缓离开枝头。它们不急不躁，只是轻轻旋转着、翻飞着，仿佛在与树枝做最后的告别，又像是在空中完成一场静谧的舞蹈。

阳光透过稀疏的枝隙洒落下来，落叶在光影之间浮动，忽明忽暗，像燃尽前最后的温暖。它们有的划出长长的弧线，轻轻落在溪水边，随水流缓缓漂远；有的则停在石阶、屋檐或泥土上，层层堆叠，铺成柔软的红色地毯。踩上去时，会发出细碎的声响，那是秋天最轻柔的回应。

整个世界仿佛慢了下来。空气中带着干净的凉意与淡淡的草木香，远处的山峦被薄雾笼罩，显得深远而宁静。枫叶一片一片飘落，不只是季节的更替，更像时间具象化的流逝，让人不自觉放慢脚步，在这片金红交织的世界里，感受秋天最温柔的告别与沉静之美。
""",
        # 要合成的文本内容

        language="Chinese",
        # 语言类型（影响发音模型）

        speaker="Serena",
        # 说话人（音色 ID，对应模型内置或自定义音色）

        instruct="温柔,朗读",
        # 情感/语气控制（可选）
        # 不填则默认中性语气
    )

    # （新增）结束计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"⏱️ 推理耗时: {end_time - start_time:.3f} 秒")

    # ===================== 音频保存 =====================
    sf.write(
        "output_custom_voice.wav",  # 输出文件名
        wavs[0],                   # 第一条生成的音频数据（numpy array）
        sr                         # 采样率（sampling rate）
    )