import os
# 配置hf镜像，写在代码最前面，优先生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download



def download_hf_model(repo_id: str, local_dir: str):
    """
    从Hugging Face下载模型，国内镜像加速，自带断点续传
    :param repo_id: 模型仓库名称，例如 "Qwen/Qwen2-7B-Instruct"
    :param local_dir: 模型保存的本地文件夹（绝对路径）
    """
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,      # 开启断点续传，中断后再次运行继续下载
        force_download=False,      # 已经下载完成的文件不会重复下载
        local_dir_use_symlinks=False, # 不生成软链接，把文件真实存到指定文件夹
        # 可选配置：过滤文件，不需要则注释下面两行
        # allow_patterns=["*.bin", "*.safetensors", "*.json", "vocab.txt"],
        # ignore_patterns=["*.md", ".gitattributes", "*.pdf"]
    )

if __name__ == "__main__":
    # 修改这里为你的模型id和保存路径
    model_repo = "JetLM/SDAR-30B-A3B-Chat-b32"
    save_path = "/root/lkd/team/Models/SDAR-30B-A3B-Chat-b32"
    download_hf_model(model_repo, save_path)