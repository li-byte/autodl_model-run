import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


def main():
    """
    主函数：加载Qwen2.5-Omni模型，处理视频输入，生成文本和音频输出
    """
    # 默认：在可用设备上加载模型（自动选择GPU/CPU）
    # 加载预训练的Qwen2.5-Omni模型，自动选择数据类型和设备
    # model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    #     "../models",
    #     torch_dtype="auto",  # 自动选择合适的数据类型
    #     device_map="auto"  # 自动映射到可用设备
    # )

    # 建议启用flash_attention_2以获得更好的加速和内存节省
    # 如果GPU支持，可以取消下面的注释来使用flash attention 2
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "../../models/Qwen2_5_Omni_7B",
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",  # 启用flash attention 2
    )

    # 加载处理器：用于处理文本、音频、图像和视频的输入输出
    processor = Qwen2_5OmniProcessor.from_pretrained("../../models/Qwen2_5_Omni_7B")

    # 定义对话内容
    conversation = [
        {
            "role": "system",  # 系统角色：定义AI助手的身份和行为
            "content": [
                {"type": "text",
                 "text": "你是Qwen，一个由阿里巴巴集团Qwen团队开发的虚拟人，能够感知听觉和视觉输入，并生成文本和语音。"}
            ],
        },
        {
            "role": "user",  # 用户角色：提供输入内容
            "content": [
                {"type": "text",
                 "text": "写一篇春天里的文章"},
                # {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
                # # 视频URL
            ],
        },
    ]

    # 设置是否使用视频中的音频
    USE_AUDIO_IN_VIDEO = False

    # 准备推理数据
    # 应用聊天模板，将对话转换为模型可理解的文本格式
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # 处理多模态信息：从对话中提取音频、图像和视频
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # 将处理后的输入编码为模型可接受的格式
    inputs = processor(
        text=text,  # 文本输入
        audio=audios,  # 音频输入
        images=images,  # 图像输入
        videos=videos,  # 视频输入
        return_tensors="pt",  # 返回PyTorch张量格式
        padding=True,  # 自动填充到相同长度
        use_audio_in_video=USE_AUDIO_IN_VIDEO  # 是否使用视频中的音频
    )

    # 将输入数据移动到模型所在的设备，并转换为模型的数据类型
    inputs = inputs.to(model.device).to(model.dtype)

    # 推理：生成输出文本和音频
    # text_ids: 生成的文本token ID
    # audio: 生成的音频数据
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # 解码生成的文本token ID为可读的文本
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("生成的文本:", text)

    # 将生成的音频保存为WAV文件
    sf.write(
        "output.wav",  # 输出文件名
        audio.reshape(-1).detach().cpu().numpy(),  # 音频数据
        samplerate=24000  # 采样率24kHz
    )
    print("音频已保存为 output.wav")


# 程序入口点
if __name__ == "__main__":
    main()