import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # 下载 Qwen2.5-3B-Instruct-GPTQ-Int4 模型
    snapshot_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        local_dir="E:\models\Qwen2.5-3B-Instruct-GPTQ-Int4",  # 修改为对应的模型路径
        resume_download=True,
        token="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"  # 请确认这个 token 是否有效
    )

    print("✓ Qwen2.5-3B-Instruct-GPTQ-Int4 模型下载完成！")
    print("模型保存路径: models/Qwen2.5-3B-Instruct-GPTQ-Int4")