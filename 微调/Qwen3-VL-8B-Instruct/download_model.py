import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # 下载 deepseek-vl2-tiny 模型
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        local_dir="../../models/Qwen3-VL-8B-Instruct",
        resume_download=True
    )

    print("✓ Qwen/Qwen3-VL-4B-Instruct 模型下载完成！")
    print("模型保存路径: /models/Qwen/Qwen3-VL-4B-Instruct")