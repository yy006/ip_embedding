import json
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.ensemble import IsolationForest
from sklearn.tree import plot_tree

from dataclasses import dataclass, asdict
from numpy.linalg import norm

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# ============================================================
# 設定エリア
# ============================================================

USE_RUNS_CSV = True
RUNS_CSV_PATH = ARTIFACTS_ROOT / "alpha_sweep_mapping_a82g3rke.csv"

OUT_DIR_NAME = "ノルム制約各攻撃_incremental_埋め込みの寄与_knn_cosine_hybrid"

mode_single = False
SINGLE_RUNS_CSV_A = ARTIFACTS_ROOT / "alpha_sweep_mapping_osw7lu2p.csv"
SINGLE_RUNS_CSV_B = ARTIFACTS_ROOT / "alpha_sweep_mapping_txigea6j.csv"

DATASET = "UNSW-NB15"

SEED_RANGE = range(40, 51)
TEST_SIZE = 0.2

USE_TEST_OVERSAMPLING = True
MIN_ATTACK_IN_TEST = 500

USE_COLS_BASE = [
    "proto","state","dur","sbytes","dbytes","sloss","dloss","service",
    "Sload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb",
    "smeansz","trans_depth","res_bdy_len","Sjit","Djit","sttl",
]

ATTACK_LABEL = 1
BENIGN_LABEL = 0

# ============================================================
# ユーティリティ
# ============================================================

def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_embeddings(path: str | Path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)

def load_experiment_config(dataset: str, run_id: str) -> dict:
    """
    experiments/{dataset}/{run_id}/experiment.json を読み込む
    """
    exp_dir = Path("experiments") / dataset / run_id
    json_path = exp_dir / "experiment.json"
    if not json_path.exists():
        raise FileNotFoundError(f"experiment.json not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def get_embedding_interface(model_obj):
    wv = getattr(model_obj, "wv", None)
    if wv is not None:
        return wv
    embs = model_obj["model_state"]["in_embed.weight"].detach().cpu().numpy()
    token2id = model_obj["token2id"]

    class TorchKV:
        def __init__(self, v, t):
            self.vectors = v
            self.key_to_index = t
            self.vector_size = v.shape[1]
        def get_vector(self, k):
            return self.vectors[self.key_to_index[k]]
    return TorchKV(embs, token2id)

def resolve_paths_from_config_incremental(dataset: str, run_id: str) -> dict:
    """
    incremental 用:
      - blocks の最後から2番目を train ブロック
      - blocks の最後       を test ブロック
    として、
      - input_csv (test ブロックのCSV)
      - embed_train_path (trainブロックのmodel_path)
      - embed_test_path  (testブロックのmodel_path)
    を返す。
    """
    cfg = load_experiment_config(dataset, run_id)
    blocks: dict = cfg["blocks"]
    results_blocks: dict = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    if len(block_nums) < 2:
        raise ValueError(f"incremental なのにブロック数が2未満です: {block_nums}")

    train_num = block_nums[-3]
    test_num = block_nums[-2]

    train_key_blocks = str(train_num)
    test_key_blocks = str(test_num)
    train_key_results = f"{train_num:03d}"
    test_key_results = f"{test_num:03d}"

    input_csv = blocks[test_key_blocks]
    embed_train_path = results_blocks[train_key_results]["model"]["model_path"]
    embed_test_path = results_blocks[test_key_results]["model"]["model_path"]

    return {
        "input_csv": input_csv,
        "embed_train_path": embed_train_path,
        "embed_test_path": embed_test_path,
    }

def compute_mean_vector(wv):
    return wv.vectors.mean(axis=0)

def ip_to_vec(ip, wv, mean_vec):
    try:
        return wv.get_vector(ip)
    except KeyError:
        return mean_vec

def attach_srcip_embedding(df, wv, mean_vec):
    dim = wv.vectors.shape[1]
    vecs = np.vstack([ip_to_vec(ip, wv, mean_vec) for ip in df["srcip"].astype(str)])
    emb_cols = [f"src_emb_{i}" for i in range(dim)]
    return pd.concat([df, pd.DataFrame(vecs, columns=emb_cols, index=df.index)], axis=1), emb_cols

def random_oversample_minority(X, y, min_pos, seed):
    pos_idx = np.where(y.values == ATTACK_LABEL)[0]
    if len(pos_idx) >= min_pos or len(pos_idx) == 0:
        return X, y
    rng = np.random.default_rng(seed)
    extra = rng.choice(pos_idx, size=min_pos - len(pos_idx), replace=True)
    return (
        pd.concat([X, X.iloc[extra]]).reset_index(drop=True),
        pd.concat([y, y.iloc[extra]]).reset_index(drop=True),
    )

# ============================================================
# ===== 検証用関数 =====
# ============================================================

def compute_if_split_importance(pipe, feature_names):
    iso = pipe.named_steps["clf"]
    counts = np.zeros(len(feature_names), dtype=int)
    for est in iso.estimators_:
        for f in est.tree_.feature:
            if f >= 0:
                counts[f] += 1
    return pd.DataFrame({
        "feature": feature_names,
        "split_count": counts,
        "split_freq": counts / counts.sum(),
    }).sort_values("split_freq", ascending=False)

def permutation_importance_group(pipe, X, y, groups, n_repeat=5):
    base = roc_auc_score(y, -pipe.decision_function(X))
    out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeat):
            Xp = X.copy()
            for c in cols:
                Xp[c] = np.random.permutation(Xp[c].values)
            drops.append(base - roc_auc_score(y, -pipe.decision_function(Xp)))
        out[g] = {"mean_drop_auc": float(np.mean(drops))}
    return base, out

def compute_embedding_drift(df, wv_tr, wv_te, mv_tr, mv_te):
    cos, l2 = [], []
    for ip in df["srcip"].astype(str).unique():
        v1 = ip_to_vec(ip, wv_tr, mv_tr)
        v2 = ip_to_vec(ip, wv_te, mv_te)
        cos.append(1 - np.dot(v1, v2) / (norm(v1)*norm(v2) + 1e-8))
        l2.append(norm(v1 - v2))
    return {
        "cosine_diff_mean": float(np.mean(cos)),
        "l2_diff_mean": float(np.mean(l2)),
    }

# ============================================================
# グローバル
# ============================================================

df = None
wv_train = None
wv_test = None
mean_vec_train = None
mean_vec_test = None
CAT_COLS = []
NUM_COLS = []
OUT_DIR = None

# ============================================================
# 1 seed 実行
# ============================================================

def run_isoforest_for_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = ensure_outdir(OUT_DIR / f"seed_{seed}")

    X = df.drop(columns=["Label"])
    y = df["Label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    if USE_TEST_OVERSAMPLING:
        X_te, y_te = random_oversample_minority(X_te, y_te, MIN_ATTACK_IN_TEST, seed)

    X_tr, emb_cols = attach_srcip_embedding(X_tr, wv_train, mean_vec_train)
    X_te, _ = attach_srcip_embedding(X_te, wv_test, mean_vec_test)

    ALL_NUM = NUM_COLS + emb_cols

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", "passthrough", ALL_NUM),
    ])

    iso = IsolationForest(
        n_estimators=1000,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )

    pipe = Pipeline([("prep", prep), ("clf", iso)])
    pipe.fit(X_tr[y_tr == BENIGN_LABEL])

    score = -pipe.decision_function(X_te)

    # ===== 木の可視化 =====
    fnames = pipe.named_steps["prep"].get_feature_names_out()
    plt.figure(figsize=(20, 10))
    plot_tree(iso.estimators_[0], max_depth=3, feature_names=fnames)
    plt.savefig(out_dir / "isoforest_tree_0.png")
    plt.close()

    # ===== 検証① split importance =====
    df_imp = compute_if_split_importance(pipe, fnames)
    df_imp.to_csv(out_dir / "if_split_importance.csv", index=False)
    df_imp["is_embedding"] = df_imp["feature"].str.contains("src_emb_")
    df_imp.groupby("is_embedding")["split_freq"].sum().to_csv(
        out_dir / "if_split_importance_group.csv"
    )

    # ===== 検証② permutation =====
    base_auc, perm = permutation_importance_group(
        pipe,
        X_te,
        y_te,
        {"embedding": emb_cols, "non_embedding": CAT_COLS + NUM_COLS},
    )
    with open(out_dir / "permutation_importance_group.json", "w") as f:
        json.dump({"base_auc": base_auc, "groups": perm}, f, indent=2)

    # ===== 検証③ drift =====
    drift = compute_embedding_drift(df, wv_train, wv_test, mean_vec_train, mean_vec_test)
    with open(out_dir / "embedding_drift.json", "w") as f:
        json.dump(drift, f, indent=2)

    return {
        "seed": seed,
        "roc_auc": roc_auc_score(y_te, score),
        "ap": average_precision_score(y_te, score),
    }

# ===============================
# kNN (cosine) for 1 seed
# ===============================

def run_knn_cosine_for_seed(seed: int, k: int = 20):
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = ensure_outdir(OUT_DIR / f"seed_{seed}")

    # ---------------------------
    # split
    # ---------------------------
    X = df.drop(columns=["Label"])
    y = df["Label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    if USE_TEST_OVERSAMPLING:
        X_te, y_te = random_oversample_minority(
            X_te, y_te, MIN_ATTACK_IN_TEST, seed
        )

    # ---------------------------
    # attach embeddings
    # ---------------------------
    X_tr, emb_cols = attach_srcip_embedding(
        X_tr, wv_train, mean_vec_train
    )
    X_te, _ = attach_srcip_embedding(
        X_te, wv_test, mean_vec_test
    )

    ALL_NUM_COLS = NUM_COLS + emb_cols

    # ---------------------------
    # preprocessing
    #   - OneHot(cat)
    #   - passthrough(num)
    #   - L2 normalize (for cosine)
    # ---------------------------
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", "passthrough", ALL_NUM_COLS),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        [
            ("prep", preprocess),
            ("norm", Normalizer(norm="l2")),
        ]
    )

    X_tr_feat = pipe.fit_transform(X_tr[y_tr == BENIGN_LABEL])
    X_te_feat = pipe.transform(X_te)

    # ---------------------------
    # kNN (cosine)
    # ---------------------------
    knn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        n_jobs=-1,
    )
    knn.fit(X_tr_feat)

    # distances: (n_test, k)
    dists, _ = knn.kneighbors(X_te_feat, return_distance=True)

    # anomaly score = mean cosine distance
    score = dists.mean(axis=1)

    # ---------------------------
    # metrics
    # ---------------------------
    roc = roc_auc_score(y_te, score)
    ap = average_precision_score(y_te, score)

    print(f"[kNN-cosine][seed={seed}] AUC={roc:.4f} AP={ap:.4f}")

    # save
    pd.DataFrame(
        {
            "y_true": y_te.values,
            "anom_score": score,
        }
    ).to_csv(out_dir / "pred_knn_cosine.csv", index=False)

    return {
        "seed": seed,
        "roc_auc": roc,
        "ap": ap,
    }

def run_knn_hybrid_for_seed(seed: int, k: int = 20, alpha: float = 0.9):
    """
    alpha: 非embedding側スコアの重み（0.7〜0.95推奨）
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = ensure_outdir(OUT_DIR / f"seed_{seed}")

    X = df.drop(columns=["Label"])
    y = df["Label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    if USE_TEST_OVERSAMPLING:
        X_te, y_te = random_oversample_minority(X_te, y_te, MIN_ATTACK_IN_TEST, seed)

    # 埋め込み付与
    X_tr, emb_cols = attach_srcip_embedding(X_tr, wv_train, mean_vec_train)
    X_te, _ = attach_srcip_embedding(X_te, wv_test, mean_vec_test)

    # ---------------------------
    # 1) 非embedding特徴: OneHot + 数値 Standardize → Euclidean
    # ---------------------------
    non_num_cols = NUM_COLS
    non_cat_cols = CAT_COLS

    prep_non = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), non_cat_cols),
            ("num", StandardScaler(), non_num_cols),
        ],
        remainder="drop",
    )

    Xtr_non = prep_non.fit_transform(X_tr[y_tr == BENIGN_LABEL])
    Xte_non = prep_non.transform(X_te)

    knn_non = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn_non.fit(Xtr_non)
    d_non, _ = knn_non.kneighbors(Xte_non)
    score_non = d_non.mean(axis=1)

    # ---------------------------
    # 2) embedding特徴: L2 normalize → cosine
    # ---------------------------
    prep_emb = Pipeline(
        [
            ("sel", ColumnTransformer([("emb", "passthrough", emb_cols)], remainder="drop")),
            ("norm", Normalizer(norm="l2")),
        ]
    )

    Xtr_emb = prep_emb.fit_transform(X_tr[y_tr == BENIGN_LABEL])
    Xte_emb = prep_emb.transform(X_te)

    knn_emb = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn_emb.fit(Xtr_emb)
    d_emb, _ = knn_emb.kneighbors(Xte_emb)
    score_emb = d_emb.mean(axis=1)

    # ---------------------------
    # 3) 合成スコア
    # ---------------------------
    score = alpha * score_non + (1 - alpha) * score_emb

    roc = roc_auc_score(y_te, score)
    ap = average_precision_score(y_te, score)

    print(f"[Hybrid kNN][seed={seed}] AUC={roc:.4f} AP={ap:.4f} alpha={alpha} k={k}")

    pd.DataFrame(
        {
            "y_true": y_te.values,
            "score_non": score_non,
            "score_emb": score_emb,
            "anom_score": score,
        }
    ).to_csv(out_dir / "pred_knn_hybrid.csv", index=False)

    return {"seed": seed, "roc_auc": roc, "ap": ap}

# ============================================================
# CSV single モード
# ============================================================

def main_from_csv_single():
    global df, wv_train, wv_test, mean_vec_train, mean_vec_test
    global CAT_COLS, NUM_COLS, OUT_DIR

    df_a = pd.read_csv(SINGLE_RUNS_CSV_A)
    df_b = pd.read_csv(SINGLE_RUNS_CSV_B)

    for idx in range(len(df_a)):
        run_id_a = str(df_a.iloc[idx]["run_id"])
        run_id_b = str(df_b.iloc[idx]["run_id"])

        cfg_a = json.load(
            open(Path("experiments") / DATASET / run_id_a / "experiment.json")
        )
        cfg_b = json.load(
            open(Path("experiments") / DATASET / run_id_b / "experiment.json")
        )
        input_csv = cfg_b["blocks"][list(cfg_b["blocks"].keys())[-1]]
        embed_train = cfg_a["results"]["blocks"][sorted(cfg_a["results"]["blocks"])[-1]]["model"]["model_path"]
        embed_test = cfg_b["results"]["blocks"][sorted(cfg_b["results"]["blocks"])[-1]]["model"]["model_path"]

        OUT_DIR = ensure_outdir(Path("eval")/OUT_DIR_NAME/f"single_{idx:03d}")

        df = load_csv(input_csv)

        model_tr = load_embeddings(embed_train)
        model_te = load_embeddings(embed_test)

        wv_train = get_embedding_interface(model_tr)
        wv_test = get_embedding_interface(model_te)
        mean_vec_train = compute_mean_vector(wv_train)
        mean_vec_test = compute_mean_vector(wv_test)

        use_cols = [c for c in USE_COLS_BASE if c in df.columns]
        CAT_COLS = [c for c in ["proto","state","service"] if c in use_cols]
        NUM_COLS = [c for c in use_cols if c not in CAT_COLS]

        results = []
        for s in SEED_RANGE:
            results.append(run_knn_cosine_for_seed(s, k=20))

        pd.DataFrame(results).to_csv(OUT_DIR/"seed_results.csv", index=False)

def main_from_csv_incremental():
    """
    alpha_anom,mode,run_id のCSVを読み込み、
    mode == "incremental" の行のみ run_id ごとに評価するモード
    """
    global df, wv_train, wv_test, mean_vec_train, mean_vec_test
    global CAT_COLS, NUM_COLS, OUT_DIR
    global CURRENT_RUN_ID, CURRENT_MODE, CURRENT_ALPHA

    runs_df = pd.read_csv(RUNS_CSV_PATH)
    required = {"alpha_anom", "mode", "run_id"}
    if not required.issubset(runs_df.columns):
        raise ValueError(
            f"{RUNS_CSV_PATH} に必要な列 {required} が揃っていません。現在の列: {runs_df.columns}"
        )

    # mode が incremental の行だけ使う
    runs_df = runs_df[runs_df["mode"] == "incremental"].copy()
    if runs_df.empty:
        print("[INFO] CSV内に mode==incremental の行がありません。何も実行しません。")
        return

    for _, row in runs_df.iterrows():
        alpha_anom = float(row["alpha_anom"])
        mode = str(row["mode"])  # "incremental" 確定
        run_id = str(row["run_id"])

        CURRENT_RUN_ID = run_id
        CURRENT_MODE = mode
        CURRENT_ALPHA = alpha_anom

        print(
            f"\n########## run_id={run_id}, mode={mode}, alpha_anom={alpha_anom} ##########"
        )

        paths = resolve_paths_from_config_incremental(DATASET, run_id)
        input_csv = paths["input_csv"]
        embed_train_path = paths["embed_train_path"]
        embed_test_path = paths["embed_test_path"]

        # run_id 単位で出力ディレクトリを分ける
        OUT_DIR = ensure_outdir(Path("eval") / OUT_DIR_NAME / run_id)
        print("OUT_DIR:", OUT_DIR)

        df = load_csv(input_csv)
        if "Label" not in df.columns:
            raise KeyError("Label 列が見つかりません。")
        if "srcip" not in df.columns:
            raise KeyError("srcip 列が見つかりません。")

        model_train = load_embeddings(embed_train_path)
        model_test = load_embeddings(embed_test_path)
        wv_train = get_embedding_interface(model_train)
        wv_test = get_embedding_interface(model_test)
        mean_vec_train = compute_mean_vector(wv_train)
        mean_vec_test = compute_mean_vector(wv_test)

        use_cols = [c for c in USE_COLS_BASE if c in df.columns]
        CAT_COLS = [c for c in ["proto", "state", "service"] if c in use_cols]
        NUM_COLS = [c for c in use_cols if c not in CAT_COLS]

        print("USE_COLS:", use_cols)
        print("CAT_COLS:", CAT_COLS)
        print("NUM_COLS:", NUM_COLS)

        results = []
        for s in SEED_RANGE:
            results.append(run_knn_hybrid_for_seed(s, k=20, alpha=0.5))

        pd.DataFrame(results).to_csv(OUT_DIR/"seed_results.csv", index=False)

# ============================================================
# エントリポイント
# ============================================================

if __name__ == "__main__":
    if USE_RUNS_CSV:
        if mode_single:
            main_from_csv_single()
        else:
            main_from_csv_incremental()