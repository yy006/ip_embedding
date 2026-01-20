import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# 設定
# ============================================================

RUN_ID = "2026-01-19T21-29-51_single_5q2fo8mp"
BLOCK_ID = 1
DATASET = "CIC-IDS2017"

ARTIFACTS_ROOT = Path("/workspace/experiments")
EXPERIMENT_JSON = ARTIFACTS_ROOT / DATASET / RUN_ID / "experiment.json"
OUT_DIR = ARTIFACTS_ROOT / DATASET / RUN_ID / "embed_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PCA_SCALE = 1

EXCLUDE_IPS = {
    "10.40.182.6",
    "149.171.126.12",
}

IPS_REF = {
    "205.174.165.73",
    "192.168.10.15",
    "192.168.10.8",
    "192.168.10.9",
    "192.168.10.14",
    "192.168.10.5",
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
        raise KeyError(f"BLOCK_ID={block_id} not found")
    return blocks[key]

def load_embeddings(model_path: Path):
    state = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
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

    X, token2id = load_embeddings(model_path)
    V, D = X.shape
    print(f"[INFO] embeddings loaded: vocab={V}, dim={D}")

    # --------------------------------------------------------
    # 除外IP
    # --------------------------------------------------------
    exclude_ids = [token2id[ip] for ip in EXCLUDE_IPS if ip in token2id]
    if exclude_ids:
        mask = np.ones(V, dtype=bool)
        mask[exclude_ids] = False
        X = X[mask]

        old_id2token = {v: k for k, v in token2id.items()}
        kept_old_ids = np.nonzero(mask)[0]
        token2id = {
            old_id2token[old_i]: new_i
            for new_i, old_i in enumerate(kept_old_ids)
        }

    # --------------------------------------------------------
    # 色分けマスク
    # --------------------------------------------------------
    ref_ids = {token2id[ip] for ip in IPS_REF if ip in token2id}
    is_ref = np.array([i in ref_ids for i in range(X.shape[0])])

    # ========================================================
    # 1. ノルム統計
    # ========================================================
    norms = np.linalg.norm(X, axis=1)
    stats = {
        "vocab_size": int(X.shape[0]),
        "dim": int(D),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "global_variance": float(X.var()),
    }

    with open(OUT_DIR / f"stats_block_{BLOCK_ID:03d}.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ========================================================
    # 2A. 原点基準：距離 × 角度
    # ========================================================
    eps = 1e-8
    dist0 = np.linalg.norm(X, axis=1)
    X0_norm = X / (dist0[:, None] + eps)
    dir0 = X0_norm.mean(axis=0)
    dir0 /= np.linalg.norm(dir0) + eps
    theta0 = np.arccos(np.clip(X0_norm @ dir0, -1, 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dist0[~is_ref], theta0[~is_ref], s=25, alpha=0.6, color="tab:blue")
    ax.scatter(dist0[is_ref],  theta0[is_ref],  s=25, alpha=0.6, color="tab:red")
    ax.set_xlabel("||x||")
    ax.set_ylabel("angle [rad]")
    ax.set_title("Distance × Angle (Origin-based)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"dist_angle_origin_{BLOCK_ID:03d}.png", dpi=150)
    plt.close()

    # ========================================================
    # 2B. 重心基準：距離 × 角度
    # ========================================================
    mu = X.mean(axis=0)
    Xc = X - mu
    dist = np.linalg.norm(Xc, axis=1)
    Xc_norm = Xc / (dist[:, None] + eps)
    dir_mu = Xc_norm.mean(axis=0)
    dir_mu /= np.linalg.norm(dir_mu) + eps
    theta = np.arccos(np.clip(Xc_norm @ dir_mu, -1, 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dist[~is_ref], theta[~is_ref], s=25, alpha=0.6, color="tab:blue")
    ax.scatter(dist[is_ref],  theta[is_ref],  s=25, alpha=0.6, color="tab:red")
    ax.set_xlabel("||x - μ||")
    ax.set_ylabel("angle [rad]")
    ax.set_title("Distance × Angle (Centroid-based)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"dist_angle_centroid_{BLOCK_ID:03d}.png", dpi=150)
    plt.close()

    # ========================================================
    # 2C. 極座標
    # ========================================================
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.scatter(theta[~is_ref], dist[~is_ref], s=20, alpha=0.6, color="tab:blue")
    ax.scatter(theta[is_ref],  dist[is_ref],  s=20, alpha=0.6, color="tab:red")
    ax.set_title("Polar: Distance × Direction")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"polar_{BLOCK_ID:03d}.png", dpi=150)
    plt.close()

    # ========================================================
    # 3A. PCA
    # ========================================================
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X) * PCA_SCALE

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_pca[~is_ref, 0], X_pca[~is_ref, 1], s=30, alpha=0.6, color="tab:blue")
    ax.scatter(X_pca[is_ref,  0], X_pca[is_ref,  1], s=30, alpha=0.6, color="tab:red")
    ax.set_title("PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"pca_{BLOCK_ID:03d}.png", dpi=150)
    plt.close()

    # ========================================================
    # 3B. t-SNE
    # ========================================================
    tsne = TSNE(
        n_components=2,
        perplexity=3,
        init="pca",
        learning_rate="auto",
        random_state=0,
        n_iter=1000,
    )
    X_tsne = tsne.fit_transform(X)
    X_tsne -= X_tsne.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_tsne[~is_ref, 0], X_tsne[~is_ref, 1], s=30, alpha=0.6, color="tab:blue")
    ax.scatter(X_tsne[is_ref,  0], X_tsne[is_ref,  1], s=30, alpha=0.6, color="tab:red")
    ax.set_title("t-SNE (2D)")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"tsne_{BLOCK_ID:03d}.png", dpi=150)
    plt.close()

    print("[INFO] all plots saved in", OUT_DIR)


# ============================================================

if __name__ == "__main__":
    analyze()
