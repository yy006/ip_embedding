import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances
import torch
import json

# ============================================================
# 設定
# ============================================================

RUNS_CSV = Path("experiments/alpha_sweep_mapping_jwiyjc78 - コピー.csv")   # ★ 入力CSV
OUT_DIR = Path("eval/knn_LOO")
OUT_DIR.mkdir(exist_ok=True)

K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ATTACK_IPS = {f"175.45.176.{i}" for i in range(4)}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# 埋め込みロード
# ============================================================

def load_embeddings(path: str | Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    vectors = state["model_state"]["in_embed.weight"].detach().cpu().numpy()
    token2id = state["token2id"]
    return vectors, token2id

def build_ip_vectors(vectors, token2id):
    ip2vec = {}
    for ip, idx in token2id.items():
        ip2vec[ip] = vectors[idx]
    return ip2vec

def resolve_paths_from_run_id(run_id: str) -> Path:
    """
    experiments/{DATASET}/{run_id}/experiment.json から
    最後の model_block の model_path を取得
    """
    exp_dir = Path("experiments/UNSW-NB15") / run_id
    json_path = exp_dir / "experiment.json"

    if not json_path.exists():
        raise FileNotFoundError(f"experiment.json not found: {json_path}")

    with open(json_path, "r") as f:
        cfg = json.load(f)

    results_blocks = cfg["results"]["blocks"]
    if not results_blocks:
        raise ValueError("results.blocks is empty")

    # 最新ブロックのモデルを使う
    last_block = max(int(k) for k in results_blocks.keys())
    embed_path = results_blocks[f"{last_block:03d}"]["model"]["model_path"]

    return Path(embed_path)


# ============================================================
# IP DataFrame
# ============================================================

def build_ip_df(ip2vec):
    rows = []
    for ip, vec in ip2vec.items():
        rows.append(
            {
                "ip": ip,
                "label": 1 if ip in ATTACK_IPS else 0,
                "vec": vec,
            }
        )
    return pd.DataFrame(rows)

# ============================================================
# kNN スコア
# ============================================================

def knn_score(query_vec, ref_vecs, k):
    dists = cosine_distances(
        query_vec.reshape(1, -1),
        ref_vecs
    )[0]
    k_eff = min(k, len(dists))
    return float(np.mean(np.partition(dists, k_eff - 1)[:k_eff]))

# ============================================================
# Group-wise LOO
# ============================================================

def group_loo_knn(df, k):
    scores = []

    for _, row in df.iterrows():
        ip = row["ip"]
        vec = row["vec"]

        ref = df[
            (df["label"] == 0) &
            (df["ip"] != ip)
        ]

        if len(ref) == 0:
            scores.append(np.nan)
            continue

        ref_vecs = np.vstack(ref["vec"].values)
        score = knn_score(vec, ref_vecs, k)
        scores.append(score)

    return np.array(scores)

# ============================================================
# メイン
# ============================================================

def main():
    runs = pd.read_csv(RUNS_CSV)

    results = []

    for _, row in runs.iterrows():
        run_id = row["run_id"]
        embed_path = resolve_paths_from_run_id(run_id)

        print(f"\n=== run_id={run_id} ===")

        vectors, token2id = load_embeddings(embed_path)
        ip2vec = build_ip_vectors(vectors, token2id)
        df_ip = build_ip_df(ip2vec)

        for k in K_LIST:
            scores = group_loo_knn(df_ip, k)
            df_ip["score"] = scores

            valid = ~np.isnan(scores)
            if valid.sum() == 0:
                auc = np.nan
            else:
                auc = roc_auc_score(
                    df_ip.loc[valid, "label"],
                    df_ip.loc[valid, "score"],
                )

            print(f"k={k:>3} AUC={auc:.6f}")

            results.append(
                {
                    "run_id": run_id,
                    "k": k,
                    "auc": auc,
                    "n_ips": len(df_ip),
                    "n_attack": int(df_ip["label"].sum()),
                }
            )

    # === 保存 ===
    res_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "knn_group_loo_results.csv"
    res_df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

if __name__ == "__main__":
    main()
