import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# 設定（ここだけ編集すればOK）
# ============================================================

RUN_ID = "2026-01-17T21-46-09_incremental_wegd3i4j"
BLOCK_ID = 7
DATASET = "UNSW-NB15"

ARTIFACTS_ROOT = Path("/workspace/experiments")
EXPERIMENT_JSON = ARTIFACTS_ROOT / DATASET / RUN_ID / "experiment.json"
OUT_DIR = ARTIFACTS_ROOT / DATASET / RUN_ID / "embed_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ノルム制約が強いので拡大必須
PCA_SCALE = 1

# ============================================================
# プロット前に除外したい IP（語彙から落とす）
# ============================================================

EXCLUDE_IPS = {
    "10.40.182.6",
    "149.171.126.12",
}


# ============================================================
# util
# ============================================================

def load_experiment_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def resolve_block_info(exp: dict, block_id: int) -> dict:
    blocks = exp["results"]["blocks"]
    key = f"{int(block_id):03d}"

    if key not in blocks:
        raise KeyError(
            f"BLOCK_ID={block_id} not found. Available={list(blocks.keys())}"
        )

    return blocks[key]

def load_embeddings(model_path: Path) -> tuple[np.ndarray, dict]:
    state = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,   # PyTorch 2.6+
    )
    W = state["model_state"]["in_embed.weight"]
    return W.detach().cpu().numpy(), state["token2id"]

# ============================================================
# main
# ============================================================

def analyze():
    print(f"[INFO] run_id={RUN_ID}, block_id={BLOCK_ID}")

    exp = load_experiment_json(EXPERIMENT_JSON)
    block = resolve_block_info(exp, BLOCK_ID)

    model_path = Path(block["model"]["model_path"])
    print("[INFO] model_path =", model_path)

    # --- embeddings ---
    X, token2id = load_embeddings(model_path)
    V, D = X.shape
    print(f"[INFO] embeddings loaded: vocab={V}, dim={D}")


    # ========================================================
    # ★ 指定 IP を語彙・埋め込みから除外
    # ========================================================

    exclude_ids = [
        token2id[ip] for ip in EXCLUDE_IPS if ip in token2id
    ]

    if exclude_ids:
        mask = np.ones(V, dtype=bool)
        mask[exclude_ids] = False

        X = X[mask]

        # 新しい token2id / id2token を再構築
        old_id2token = {v: k for k, v in token2id.items()}
        kept_old_ids = np.nonzero(mask)[0]

        token2id = {
            old_id2token[old_i]: new_i
            for new_i, old_i in enumerate(kept_old_ids)
        }

        V = X.shape[0]
        print(f"[INFO] excluded {len(exclude_ids)} IPs")
        print(f"[INFO] vocab after exclusion: {V}")
    else:
        print("[INFO] no IPs excluded")

    # ========================================================
    # 1. ノルム統計（制約確認用）
    # ========================================================

    norms = np.linalg.norm(X, axis=1)

    stats = {
        "vocab_size": V,
        "dim": D,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "global_variance": float(X.var()),
    }

    stats_path = OUT_DIR / f"stats_block_{BLOCK_ID:03d}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    id2token = {v: k for k, v in token2id.items()}  # index -> IPアドレス

    print("[INFO] stats saved:", stats_path)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ========================================================
    # 2. 重心から見た方向の角度（ノルム制約下で最重要）
    # ========================================================

    # --- 重心 ---
    mu = X.mean(axis=0)

    # --- 中心化 ---
    Xc = X - mu
    eps = 1e-8

    # --- 方向正規化 ---
    Xc_norm = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + eps)

    # --- 主方向 ---
    dir_mu = Xc_norm.mean(axis=0)
    dir_mu /= (np.linalg.norm(dir_mu) + eps)

    # --- 角度 ---
    cos_theta = Xc_norm @ dir_mu
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)   # [rad]

    # --- 距離（★ここを追加） ---
    dist = np.linalg.norm(Xc, axis=1)


    # --- 統計表示 ---
    print(
        "[INFO] angle stats (rad): "
        f"mean={theta.mean():.4f}, "
        f"std={theta.std():.4f}, "
        f"min={theta.min():.4f}, "
        f"max={theta.max():.4f}"
    )

    # --- ヒストグラム ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(theta, bins=30)
    ax.set_title(f"Angle from dominant direction (block {BLOCK_ID})")
    ax.set_xlabel("angle [rad]")
    ax.set_ylabel("count")

    out = OUT_DIR / f"angle_from_direction_block_{BLOCK_ID:03d}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("[INFO] angle hist saved:", out)

    # ========================================================
    # 2.A 距離 × 角度（半径 × 方向）の2軸プロット【最重要】
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(dist, theta, s=30, alpha=0.7)

    # IPアドレスのラベルを追加
    for i in range(len(dist)):
        ip = id2token.get(i, f"id={i}")
        ax.text(dist[i], theta[i], ip, fontsize=8, alpha=0.8)

    ax.set_title(
        f"Radius vs Angle (block {BLOCK_ID})\n"
        "distance = ||e - μ||, angle = direction deviation"
    )
    ax.set_xlabel("distance from centroid ||e - μ||")
    ax.set_ylabel("angle from dominant direction [rad]")

    out = OUT_DIR / f"radius_vs_angle_block_{BLOCK_ID:03d}_with_labels.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("[INFO] radius vs angle plot with labels saved:", out)

    # ========================================================
    # 3A. PCA（大域構造）
    # ========================================================

    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X)

    # 中心化（見やすさ・比較用）
    X_pca -= X_pca.mean(axis=0)
    X_pca *= PCA_SCALE

    id2token = {v: k for k, v in token2id.items()}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=30)

    for i in range(X_pca.shape[0]):
        ip = id2token.get(i, f"id={i}")
        ax.text(X_pca[i, 0], X_pca[i, 1], ip, fontsize=8, alpha=0.8)

    ax.set_title(
        f"PCA (2D, centered ×{PCA_SCALE})\n"
        f"run={RUN_ID}, block={BLOCK_ID}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    out = OUT_DIR / f"pca_block_{BLOCK_ID:03d}_with_labels.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("[INFO] PCA plot with labels saved:", out)


    # ========================================================
    # 3B. t-SNE（局所構造）
    # ========================================================

    tsne = TSNE(
        n_components=2,
        perplexity=3,
        learning_rate="auto",
        init="pca",          # ★ PCA 初期化（重要）
        random_state=0,
        n_iter=1000,
    )

    X_tsne = tsne.fit_transform(X)

    # t-SNE は絶対位置に意味がないので中心化（任意だが推奨）
    X_tsne -= X_tsne.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30)

    for i in range(X_tsne.shape[0]):
        ip = id2token.get(i, f"id={i}")
        ax.text(X_tsne[i, 0], X_tsne[i, 1], ip, fontsize=8, alpha=0.8)

    ax.set_title(
        f"t-SNE (2D, PCA init)\n"
        f"run={RUN_ID}, block={BLOCK_ID}"
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")

    out = OUT_DIR / f"tsne_block_{BLOCK_ID:03d}_with_labels.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("[INFO] t-SNE plot with labels saved:", out)


# ============================================================

if __name__ == "__main__":
    analyze()
