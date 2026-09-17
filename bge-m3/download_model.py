import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # 下载 deepseek-vl2-tiny 模型
    snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir="../../models/BAAI_bge_m3",
        resume_download=True
    )

    print("✓ BAAI/bge-m3 模型下载完成！")
    print("模型保存路径: /BAAI_bge_m3")