import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # 下载 deepseek-vl2-tiny 模型
    snapshot_download(
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        local_dir="../models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        resume_download=True,
        token=os.getenv("HF_TOKEN")  # 从环境变量读取，避免硬编码泄露
    )

    print("✓ Qwen3-TTS-12Hz-1.7B-CustomVoice 模型下载完成！")
    print("模型保存路径: /model/Qwen3-TTS-12Hz-1.7B-CustomVoice")