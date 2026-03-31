import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # 下载 deepseek-vl2-tiny 模型
    snapshot_download(
        repo_id="Qwen/Qwen2.5-Omni-7B",
        local_dir="../models",
        resume_download=True,
        token="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"
    )

    print("✓ DeepSeek-VL2-Tiny 模型下载完成！")
    print("模型保存路径: /deepseek-vl2-tiny")