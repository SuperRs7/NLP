from huggingface_hub import snapshot_download

# 设置下载目录
local_dir = "/root/workspace/happy_llm/datasets/BelleGroup"

# 下载数据集
snapshot_download(repo_id="BelleGroup/train_3.5M_CN", repo_type="dataset", local_dir=local_dir)