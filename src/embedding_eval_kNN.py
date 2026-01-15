import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score

from config import DATASET

# ============================================================
# 基本設定
# ============================================================

#RUNS_CSV_PATH = Path("experiments/alpha_sweep_mapping_xuea63i4.csv")
RUNS_CSV_PATH = Path("experiments/alpha_sweep_mapping_jwiyjc78 - コピー.csv")
OUT_DIR = Path("eval/knn_ip_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [1, 3, 5, 10, 20]

ATTACK_LABEL = 1
BENIGN_LABEL = 0


# ============================================================
# ユーティリティ
# ============================================================

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def load_embeddings(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def get_embedding_interface(model_obj):
    if hasattr(model_obj, "wv"):
        return model_obj.wv

    embs = model_obj["model_state"]["in_embed.weight"].cpu().numpy()
    token2id = model_obj["token2id"]

    class Wv:
        def __init__(self, vectors, token2id):
            self.vectors = vectors
            self.key_to_index = token2id

        def get_vector(self, key):
            return self.vectors[self.key_to_index[key]]

    return Wv(embs, token2id)


def compute_mean_vector(wv):
    return wv.vectors.mean(axis=0)


def ip_to_vec(ip, wv, mean_vec):
    try:
        return wv.get_vector(ip)
    except KeyError:
        return mean_vec


# ============================================================
# experiment.json からパス解決
# ============================================================

def resolve_incremental_paths(run_id):
    cfg = json.load(open(f"experiments/{DATASET}/{run_id}/experiment.json"))

    blocks = cfg["blocks"]
    results = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    train_num = block_nums[-3]   # B4
    test_num  = block_nums[-2]   # B5

    return {
        "train_csv": blocks[str(train_num)],
        "test_csv":  blocks[str(test_num)],
        "embed_train": results[f"{train_num:03d}"]["model"]["model_path"],
        "embed_test":  results[f"{test_num:03d}"]["model"]["model_path"],
    }


def resolve_single_paths(run_id):
    cfg = json.load(open(f"experiments/{DATASET}/{run_id}/experiment.json"))

    blocks = cfg["blocks"]
    results = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    model_nums = sorted(int(k) for k in results.keys())

    return {
        "train_csv": blocks[str(block_nums[-2])],
        "test_csv":  blocks[str(block_nums[-1])],
        "embed_train": results[f"{model_nums[-1]:03d}"]["model"]["model_path"],
        "embed_test":  results[f"{model_nums[-1]:03d}"]["model"]["model_path"],
    }


# ============================================================
# kNN 異常スコア（IP単位）
# ============================================================

def knn_scores_ip(X_train, X_test, k):
    nn = NearestNeighbors(metric="cosine", n_neighbors=k)
    nn.fit(X_train)
    dist, _ = nn.kneighbors(X_test)
    return dist.mean(axis=1)


# ============================================================
# 1 run_id 評価
# ============================================================

def evaluate_run(run_id, mode):
    if mode == "incremental":
        paths = resolve_incremental_paths(run_id)
    elif mode == "single":
        paths = resolve_single_paths(run_id)
    else:
        raise ValueError(mode)

    df_train = load_csv(paths["train_csv"])
    df_test  = load_csv(paths["test_csv"])

    # --- IP単位に集約 ---
    df_train_ip = df_train.groupby("srcip")["Label"].max().reset_index()
    df_test_ip  = df_test.groupby("srcip")["Label"].max().reset_index()

    # --- 埋め込み ---
    wv_train = get_embedding_interface(load_embeddings(paths["embed_train"]))
    wv_test  = get_embedding_interface(load_embeddings(paths["embed_test"]))

    mean_vec_train = compute_mean_vector(wv_train)
    mean_vec_test  = compute_mean_vector(wv_test)

    # --- train: Benign IP のみ ---
    train_ips = df_train_ip[df_train_ip["Label"] == BENIGN_LABEL]["srcip"]
    X_train = np.vstack([
        ip_to_vec(ip, wv_train, mean_vec_train)
        for ip in train_ips.astype(str)
    ])

    # --- test ---
    X_test = np.vstack([
        ip_to_vec(ip, wv_test, mean_vec_test)
        for ip in df_test_ip["srcip"].astype(str)
    ])
    y_test = df_test_ip["Label"].values

    records = []
    for k in K_LIST:
        score = knn_scores_ip(X_train, X_test, k)

        auc = roc_auc_score(y_test, score)
        ap  = average_precision_score(y_test, score)

        records.append({
            "run_id": run_id,
            "mode": mode,
            "k": k,
            "roc_auc": auc,
            "ap": ap,
            "n_train_ip": len(X_train),
            "n_test_ip": len(X_test),
        })

        print(f"[{run_id}] k={k} AUC={auc:.4f} AP={ap:.4f}")

    return records


# ============================================================
# メイン
# ============================================================

def main():
    runs_df = pd.read_csv(RUNS_CSV_PATH)

    all_records = []
    for _, row in runs_df.iterrows():
        run_id = row["run_id"]
        mode   = row["mode"]

        print(f"\n===== run_id={run_id} mode={mode} =====")
        recs = evaluate_run(run_id, mode)
        all_records.extend(recs)

    out_csv = OUT_DIR / "knn_ip_results.csv"
    pd.DataFrame(all_records).to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
