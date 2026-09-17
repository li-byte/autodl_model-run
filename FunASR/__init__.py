import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
if __name__ == '__main__':

    model_dir = snapshot_download(
        repo_id="funasr/paraformer-zh",
        local_dir="./models/paraformer-zh",
        resume_download=True,
    )
    print(f"模型路径: {model_dir}")