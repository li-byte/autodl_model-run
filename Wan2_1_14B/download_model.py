import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # 下载 deepseek-vl2-tiny 模型
    snapshot_download(
        # repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        repo_id="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        local_dir="../../models/wan2_1-1_3b",
        resume_download=True,
        token=os.getenv("HF_TOKEN")  # 从环境变量读取，避免硬编码泄露
    )

    print("✓ Wan2.1-T2V-1.3B-Diffusers 模型下载完成！")
    print("模型保存路径: /models/wan2_1-1_3b")