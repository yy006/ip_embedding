
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

# 埋め込みの読み込み
def load_embeddings_torch(path: str | Path):
    p = Path(path) 

    return torch.load(p, map_location="cpu", weights_only=False)

def load_embeddings_gensim(path: str | Path):
    p = Path(path)
    with open(p, "rb") as f:
        obj = pickle.load(f)

    return obj

def get_embedding_interface(model_obj):
    # gensim のときはそのまま返す
    wv = getattr(model_obj, "wv", None)
    if wv is not None:
        return wv

    # PyTorch 版 (state_dict + token2id) からラッパを作る
    embs = model_obj["model_state"]["in_embed.weight"].detach().cpu().numpy()
    token2id = model_obj["token2id"]

    class TorchKeyedVectorsLike:
        def __init__(self, vectors, token2id):
            self.vectors = vectors
            self.vector_size = vectors.shape[1]
            self.key_to_index = token2id
            self.index_to_key = list(token2id.keys())

        def get_vector(self, key: str):
            return self.vectors[self.key_to_index[key]]

        def __getitem__(self, key: str):
            return self.get_vector(key)

    return TorchKeyedVectorsLike(embs, token2id)

GENSIM_MODEL = "/workspace/experiments/UNSW-NB15/2025-12-05T07-18-50_incremental_q0ovon3w/models/model_block_004"
TORCH_MODEL  = "/workspace/experiments/UNSW-NB15/2025-12-05T07-57-47_incremental_urxjaak9/models/model_block_004"

gensim_model = load_embeddings_gensim(GENSIM_MODEL)
torch_model  = load_embeddings_torch(TORCH_MODEL)

wv_gen  = get_embedding_interface(gensim_model)
wv_torch = get_embedding_interface(torch_model)

# 統計量を計算する関数
def describe_vectors(name, vecs):
    # 各次元の平均・標準偏差
    dim_means = vecs.mean(axis=0)   # shape: (D,)
    dim_stds  = vecs.std(axis=0)    # shape: (D,)

    # 各ベクトルの L2 ノルム
    l2 = np.linalg.norm(vecs, axis=1)  # shape: (V,)

    print(f"\n=== {name} ===")
    print(f" vocab_size = {vecs.shape[0]}, dim = {vecs.shape[1]}")
    print(f" dim_means: mean={dim_means.mean():.4f}, std_of_means={dim_means.std():.4f}")
    print(f" dim_stds : mean={dim_stds.mean():.4f}, std_of_stds={dim_stds.std():.4f}")
    print(f" L2 norms : mean={l2.mean():.4f}, std={l2.std():.4f}, "
          f"min={l2.min():.4f}, max={l2.max():.4f}")

    return {
        "dim_means": dim_means,
        "dim_stds": dim_stds,
        "l2": l2,
    }

# 実行部分

stats_gen   = describe_vectors("gensim", wv_gen.vectors)
stats_torch = describe_vectors("torch",  wv_torch.vectors)

# === 図の保存先ディレクトリ ===
OUT_DIR = Path("vec_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# L2 ノルム分布のヒストグラム
plt.figure(figsize=(8,4))
plt.hist(stats_gen["l2"],   bins=50, alpha=0.5, label="gensim")
plt.hist(stats_torch["l2"], bins=50, alpha=0.5, label="torch")
plt.xlabel("L2 norm")
plt.ylabel("count")
plt.title("L2 norm distribution (full vocab)")
plt.legend()
plt.tight_layout()

l2_path = OUT_DIR / "l2_norm_hist.png"
plt.savefig(l2_path, dpi=180, bbox_inches="tight")
plt.close()
print("L2 norm 図を保存しました:", l2_path)

# 各次元の標準偏差を並べて比較
plt.figure(figsize=(10,4))
plt.plot(stats_gen["dim_stds"],   label="gensim dim std")
plt.plot(stats_torch["dim_stds"], label="torch dim std")
plt.xlabel("dimension index")
plt.ylabel("std of that dimension")
plt.title("Per-dimension std (full vocab)")
plt.legend()
plt.tight_layout()

std_path = OUT_DIR / "dim_std_compare.png"
plt.savefig(std_path, dpi=180, bbox_inches="tight")
plt.close()