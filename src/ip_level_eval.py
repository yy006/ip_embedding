#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IP-level embedding evaluation with:
- STEP-based stdout logging
- incremental / batch / single support
- join with config CSV
- attack × Radius mean AUC/AP plots

研究用・論文用にそのまま使える評価スクリプト。
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score

from config import *  # ARTIFACTS_ROOT など

# ============================================================
# 設定
# ============================================================

DATASET = "UNSW-NB15"

MODE_SINGLE = False
MODE_INCREMENTAL = True

RUNS_CSV_PATH = ARTIFACTS_ROOT / "alpha_sweep_mapping_a82g3rke.csv"

OUT_DIR = Path("eval/ip_level_embedding_eval_summary")

KNN_K = 10
LOF_K = 20

FLOW_IP_COL = "srcip"
FLOW_LABEL_COL = "Label"
ATTACK_LABEL = 1

VERBOSE = True

# ============================================================
# logging utilities
# ============================================================

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_step(step: int, title: str):
    if VERBOSE:
        print(f"\n[{ts()}] [STEP {step}] {title}")
        print("-" * (len(title) + 12))

def log_info(msg: str):
    if VERBOSE:
        print(f"[{ts()}]   {msg}")

def log_stat(name: str, value):
    if VERBOSE:
        print(f"[{ts()}]     - {name}: {value}")

# ============================================================
# common utils
# ============================================================

def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_embeddings(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)

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

def compute_mean_vector(wv) -> np.ndarray:
    return wv.vectors.mean(axis=0)

def ip_to_vec(ip: str, wv, mean_vec: np.ndarray) -> np.ndarray:
    try:
        return wv.get_vector(ip)
    except KeyError:
        return mean_vec

# ============================================================
# IP-level dataset
# ============================================================

def build_ip_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ip_df = (
        df.groupby(FLOW_IP_COL)[FLOW_LABEL_COL]
        .max()
        .reset_index()
        .rename(columns={FLOW_LABEL_COL: "y"})
    )
    ip_df["y"] = (ip_df["y"].astype(int) == ATTACK_LABEL).astype(int)
    return ip_df

def ip_df_to_Xy(ip_df, wv, mean_vec):
    X = np.vstack([
        ip_to_vec(ip, wv, mean_vec)
        for ip in ip_df[FLOW_IP_COL].astype(str)
    ])
    y = ip_df["y"].values
    return X, y

# ============================================================
# anomaly scores
# ============================================================

def knn_score_cosine(X_train, X_test, k):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_train)
    dist, _ = nn.kneighbors(X_test)
    return dist.mean(axis=1)

def lof_score_cosine(X_train, X_test, k):
    lof = LocalOutlierFactor(n_neighbors=k, metric="cosine", novelty=True)
    lof.fit(X_train)
    return -lof.decision_function(X_test)

# ============================================================
# one run evaluation
# ============================================================

def run_ip_level_eval(df_flow, wv_tr, wv_te):
    log_step(1, "build IP-level dataset")
    ip_df = build_ip_level_dataset(df_flow)
    log_stat("n_ip_total", len(ip_df))
    log_stat("attack_ip_ratio", ip_df["y"].mean())

    log_step(2, "split train/test (normal-only train)")
    ip_train = ip_df[ip_df["y"] == 0]
    ip_test = ip_df
    log_stat("n_ip_train", len(ip_train))
    log_stat("n_ip_test", len(ip_test))

    log_step(3, "vectorize IPs")
    X_tr, _ = ip_df_to_Xy(ip_train, wv_tr, compute_mean_vector(wv_tr))
    X_te, y_te = ip_df_to_Xy(ip_test,  wv_te, compute_mean_vector(wv_te))
    log_stat("feature_dim", X_tr.shape[1])

    log_step(4, "compute anomaly scores (kNN / LOF)")
    score_knn = knn_score_cosine(X_tr, X_te, KNN_K)
    score_lof = lof_score_cosine(X_tr, X_te, LOF_K)

    log_step(5, "evaluate metrics")
    knn_auc = roc_auc_score(y_te, score_knn)
    knn_ap  = average_precision_score(y_te, score_knn)
    lof_auc = roc_auc_score(y_te, score_lof)
    lof_ap  = average_precision_score(y_te, score_lof)

    log_info(f"kNN AUC={knn_auc:.4f}, AP={knn_ap:.4f}")
    log_info(f"LOF AUC={lof_auc:.4f}, AP={lof_ap:.4f}")

    return {
        "n_ip_train": len(ip_train),
        "n_ip_test": len(ip_test),
        "knn_auc": knn_auc,
        "knn_ap": knn_ap,
        "lof_auc": lof_auc,
        "lof_ap": lof_ap,
    }

# ============================================================
# path resolver
# ============================================================

def resolve_incremental_paths(dataset, run_id):
    cfg = json.load(open(Path("experiments")/dataset/run_id/"experiment.json"))
    blocks = cfg["blocks"]
    results = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    train_b = block_nums[-3]
    test_b = block_nums[-2]

    input_csv = Path(blocks[str(test_b)])
    emb_tr = Path(results[f"{train_b:03d}"]["model"]["model_path"])
    emb_te = Path(results[f"{test_b:03d}"]["model"]["model_path"])

    return input_csv, emb_tr, emb_te

# ============================================================
# summary & plot
# ============================================================

def summarize_and_plot(results_csv: Path, config_csv: Path, out_dir: Path):
    log_step(6, "summarize & plot (attack × Radius mean)")

    df_res = pd.read_csv(results_csv)
    df_cfg = pd.read_csv(config_csv)

    df = df_res.merge(df_cfg, on="run_id", how="inner")

    summary = (
        df.groupby(["attack", "Radius"])
        .agg(
            knn_auc_mean=("knn_auc", "mean"),
            lof_auc_mean=("lof_auc", "mean"),
            knn_ap_mean=("knn_ap", "mean"),
            lof_ap_mean=("lof_ap", "mean"),
            n=("run_id", "count"),
        )
        .reset_index()
    )

    summary.to_csv(out_dir/"summary_mean.csv", index=False)

    for metric in ["knn_auc_mean", "lof_auc_mean"]:
        plt.figure(figsize=(6,4))
        for attack, g in summary.groupby("attack"):
            plt.plot(g["Radius"], g[metric], marker="o", label=attack)
        plt.xlabel("Radius")
        plt.ylabel(metric)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"{metric}_by_attack.png")
        plt.close()

    log_info("summary & plots saved")

# ============================================================
# main
# ============================================================

def main_incremental(batch_mode=False):
    out_root = ensure_outdir(OUT_DIR / ("batch" if batch_mode else "incremental"))
    runs = load_csv(RUNS_CSV_PATH)

    all_results = []

    for _, r in runs.iterrows():
        run_id = str(r["run_id"])
        log_step(0, f"run_id={run_id}")

        csv_path, emb_tr, emb_te = resolve_incremental_paths(DATASET, run_id)
        if batch_mode:
            emb_te = emb_tr

        df = load_csv(csv_path)
        wv_tr = get_embedding_interface(load_embeddings(emb_tr))
        wv_te = get_embedding_interface(load_embeddings(emb_te))

        res = run_ip_level_eval(df, wv_tr, wv_te)
        res["run_id"] = run_id
        all_results.append(res)

    out_csv = out_root/"results.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)

    summarize_and_plot(out_csv, RUNS_CSV_PATH, out_root)

if __name__ == "__main__":
    if MODE_INCREMENTAL:
        main_incremental(batch_mode=False)
        main_incremental(batch_mode=True)
