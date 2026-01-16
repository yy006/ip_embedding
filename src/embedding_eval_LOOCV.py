#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IP-level embedding evaluation (embedding-only).

- Loads trained embeddings for a given run_id (single or incremental)
- Builds IP-level labels (any-attack) from the corresponding CSV
- Computes:
  - Prototype score (distance from benign centroid) with LOOCV scoring
  - kNN score (neighbor attack ratio) with LOOCV scoring
- Produces:
  - Run-level metrics (AUC/AP, Recall@#attack) saved to CSV
  - IP-level table per run (rank, success/failure) saved to CSV

Edit only the "設定エリア" below.
"""

# ============================================================
# imports
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 設定エリア（ここだけいじればOK）
# ============================================================

ARTIFACTS_ROOT = Path("/workspace")

# --- CSVモード / 手書きモード ---
USE_RUNS_CSV = True

# incremental の run_id 一覧（incremental時のみ参照）
RUNS_CSV_PATH = ARTIFACTS_ROOT / "experiments/alpha_sweep_mapping_xuea63i4.csv"

# single の run_id 一覧（single時のみ参照）
SINGLE_RUNS_CSV_B = ARTIFACTS_ROOT / "experiments/alpha_sweep_mapping_jwiyjc78.csv"

OUT_DIR_NAME = "eval/LOOCV_12dupfalse_sin"

# --- 実験モード ---
MODE_SINGLE = True     # True: single, False: incremental
DATASET = "UNSW-NB15"

# --- 手書きモード（USE_RUNS_CSV=False のときだけ使用） ---
MANUAL_EXPERIMENT = "2026-01-15T18-58-33_incremental_kaen3rpw"

# 手書きモードで使う埋め込みパス（必要に応じて編集）
MANUAL_EMBED_PKL_TEST = (
    f"/workspace/experiments/{DATASET}/{MANUAL_EXPERIMENT}/models/model_block_005"
)

# 手書きモードで使うCSV（ラベル生成に使う）
MANUAL_TEST_CSV = f"/workspace/datasets/{DATASET}/test.csv"

# --- 評価設定 ---
KNN_K_LIST = (1, 2, 3)   # 例: (1,3,5)
KNN_IP_TABLE_K = 2    # IP別の成功/失敗テーブルは k=3 を代表として出す
DEVICE = "cpu"

# --- ラベル設定 ---
ATTACK_LABEL = 1
BENIGN_LABEL = 0


# ============================================================
# experiment config loader / path resolvers
# ============================================================

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


def resolve_paths_from_config_incremental(dataset: str, run_id: str) -> dict:
    """
    incremental 用:
      - blocks の最後から3番目を train ブロック
      - blocks の最後から2番目を test ブロック
    として、
      - train_csv, test_csv
      - embed_train_path, embed_test_path
    を返す。
    """
    cfg = load_experiment_config(dataset, run_id)
    blocks = cfg["blocks"]
    results_blocks = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    if len(block_nums) < 3:
        raise ValueError("incremental で B4/B5 を使うには最低 3 block 必要")

    train_num = block_nums[-3]
    test_num = block_nums[-2]

    return {
        "train_csv": blocks[str(train_num)],
        "test_csv": blocks[str(test_num)],
        "embed_train_path": results_blocks[f"{train_num:03d}"]["model"]["model_path"],
        "embed_test_path": results_blocks[f"{test_num:03d}"]["model"]["model_path"],
    }


def resolve_paths_from_config_single(dataset: str, run_id: str) -> dict:
    """
    single 用:
      - csv: blocks の最後（攻撃が含まれる側だと仮定）
      - embed_path: results.blocks のうち最大の model
    """
    cfg = load_experiment_config(dataset, run_id)
    blocks: dict = cfg["blocks"]
    results_blocks: dict = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    if len(block_nums) < 1:
        raise ValueError(f"single なのにブロックが空です: {block_nums}")

    model_block_nums = sorted(int(k) for k in results_blocks.keys())
    if not model_block_nums:
        raise ValueError("results.blocks に model が1つも存在しません")

    model_num = model_block_nums[-1]
    embed_path = results_blocks[f"{model_num:03d}"]["model"]["model_path"]

    return {
        "csv": blocks[str(block_nums[-1])],
        "embed_path": embed_path,
    }


# ============================================================
# embedding loader
# ============================================================

def load_ip_embeddings(model_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Returns
    -------
    X   : np.ndarray (n_ip, D)
    ips : list[str]
    """
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if "model_state" not in state or "token2id" not in state:
        raise ValueError(f"Unexpected torch state keys: {list(state.keys())}")

    W = state["model_state"]["in_embed.weight"].detach().cpu().numpy()
    token2id = state["token2id"]

    id2ip = {i: ip for ip, i in token2id.items()}
    ips = [id2ip[i] for i in range(len(id2ip))]

    return W, ips


# ============================================================
# labels (any-attack)
# ============================================================

def _infer_ip_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to infer (ip_col, label_col) from common patterns.

    Returns
    -------
    (ip_col, label_col)
    """
    ip_candidates = ["srcip", "src_ip", "source_ip", "ip", "src_ipaddr", "SrcIP", "Src Ip", "src"]
    label_candidates = ["Label", "label", "attack", "is_attack", "any_attack", "y", "Y"]

    ip_col = next((c for c in ip_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if ip_col is None or label_col is None:
        raise ValueError(
            f"CSV must contain an IP column and a label column.\n"
            f"columns={df.columns.tolist()}\n"
            f"ip_candidates={ip_candidates}\n"
            f"label_candidates={label_candidates}"
        )
    return ip_col, label_col


def _normalize_label_to_binary(series: pd.Series) -> pd.Series:
    """
    Convert various label formats into 0/1.
    - If numeric: treat != BENIGN_LABEL as attack (any-attack)
    - If string: treat common benign strings as benign, otherwise attack
    """
    if pd.api.types.is_numeric_dtype(series):
        return (series != BENIGN_LABEL).astype(int)

    s = series.astype(str).str.lower().str.strip()
    benign_tokens = {"0", "benign", "normal", "false", "no", "none"}
    return (~s.isin(benign_tokens)).astype(int)


def make_ip_labels(csv_path: Path, ips: List[str]) -> np.ndarray:
    """
    CSV -> IP-level any-attack labels aligned to `ips` list.

    Returns
    -------
    y : np.ndarray (n_ip,) 0/1
    """
    df = pd.read_csv(csv_path)
    ip_col, label_col = _infer_ip_label_columns(df)

    # flow-level -> binary attack indicator
    flow_attack = _normalize_label_to_binary(df[label_col])

    ip2label: Dict[str, int] = {}
    for ip, idxs in df.groupby(ip_col).groups.items():
        # any-attack
        ip2label[ip] = int(flow_attack.loc[list(idxs)].any())

    return np.array([ip2label.get(ip, 0) for ip in ips], dtype=int)


# ============================================================
# scoring (LOOCV style)
# ============================================================

def proto_loocv_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Prototype score:
      score(i) = 1 - cos(x_i, mean_benign(train))
    with leave-one-out for mean computation.

    Returns
    -------
    scores : (n_ip,) higher = more anomalous
    """
    n = len(X)
    scores = np.zeros(n, dtype=float)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        x_test = X[i:i+1]

        benign_mask = (y_train == BENIGN_LABEL)
        if benign_mask.sum() == 0:
            # no benign in train -> cannot define; fallback to 0
            scores[i] = 0.0
            continue

        mu = X_train[benign_mask].mean(axis=0, keepdims=True)
        scores[i] = 1 - cosine_similarity(x_test, mu)[0, 0]

    return scores


def knn_loocv_scores(X: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    """
    kNN score:
      score(i) = mean(y of k nearest neighbors in train)
    with LOOCV split.

    Returns
    -------
    scores : (n_ip,) higher = more anomalous
    """
    n = len(X)
    scores = np.zeros(n, dtype=float)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        x_test = X[i:i+1]

        nn = NearestNeighbors(n_neighbors=min(k, len(X_train)), metric="cosine")
        nn.fit(X_train)
        _, idx = nn.kneighbors(x_test)
        scores[i] = float(y_train[idx[0]].mean())

    return scores


# ============================================================
# metrics + per-IP table
# ============================================================

def safe_auc_ap(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    If only one class present, roc_auc_score / AP may fail.
    Return nan in that case.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def summarize_ip_results(ips: List[str], y: np.ndarray, scores: np.ndarray, score_name: str) -> pd.DataFrame:
    """
    Rank-based per-IP summary with success/failure using Recall@N_attack criterion.

    Success definition:
      - attack IP: rank <= N_attack
      - benign IP: rank >  N_attack

    Returns a DataFrame sorted by rank (1 is most anomalous).
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)

    n_attack = int(y.sum())
    order = np.argsort(-scores)  # descending

    rows = []
    for rank, idx in enumerate(order, start=1):
        ip = ips[idx]
        label = int(y[idx])

        if label == ATTACK_LABEL:
            success = (rank <= n_attack)
        else:
            success = (rank > n_attack)

        rows.append({
            "ip": ip,
            "true_label": label,
            score_name: float(scores[idx]),
            "rank": rank,
            "success": bool(success),
        })

    return pd.DataFrame(rows)


def recall_at_num_attack(y: np.ndarray, scores: np.ndarray) -> float:
    """
    Recall@N_attack computed from scores:
      top N_attack entries are predicted as attack.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)

    n_attack = int(y.sum())
    if n_attack == 0:
        return float("nan")
    order = np.argsort(-scores)
    topk = order[:n_attack]
    return float((y[topk] == ATTACK_LABEL).sum() / n_attack)


# ============================================================
# collect jobs
# ============================================================

def collect_jobs() -> List[dict]:
    jobs: List[dict] = []

    if MODE_SINGLE:
        # single
        if USE_RUNS_CSV:
            df = pd.read_csv(SINGLE_RUNS_CSV_B)
            if "run_id" not in df.columns:
                raise ValueError(f"run_id column not found in {SINGLE_RUNS_CSV_B}")
            df["run_id"] = df["run_id"].astype(str)

            for _, row in df.iterrows():
                run_id = row["run_id"]
                paths = resolve_paths_from_config_single(DATASET, run_id)
                jobs.append({
                    "mode": "single",
                    "run_id": run_id,
                    "test_csv": paths["csv"],
                    "embed_test_path": paths["embed_path"],
                })
        else:
            jobs.append({
                "mode": "single",
                "run_id": MANUAL_EXPERIMENT,
                "test_csv": MANUAL_TEST_CSV,
                "embed_test_path": MANUAL_EMBED_PKL_TEST,
            })

    else:
        # incremental
        if USE_RUNS_CSV:
            df = pd.read_csv(RUNS_CSV_PATH)
            if "run_id" not in df.columns:
                raise ValueError(f"run_id column not found in {RUNS_CSV_PATH}")
            df["run_id"] = df["run_id"].astype(str)

            for _, row in df.iterrows():
                run_id = row["run_id"]
                paths = resolve_paths_from_config_incremental(DATASET, run_id)
                jobs.append({
                    "mode": "incremental",
                    "run_id": run_id,
                    "test_csv": paths["test_csv"],
                    "embed_test_path": paths["embed_test_path"],
                })
        else:
            jobs.append({
                "mode": "incremental",
                "run_id": MANUAL_EXPERIMENT,
                "test_csv": MANUAL_TEST_CSV,
                "embed_test_path": MANUAL_EMBED_PKL_TEST,
            })

    return jobs


# ============================================================
# main
# ============================================================

def main():
    out_dir = ARTIFACTS_ROOT / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs()
    print(f"[INFO] #jobs = {len(jobs)}")
    if not jobs:
        print("[WARN] no jobs. check your settings.")
        return

    run_records = []
    ip_records_paths = []

    for job in jobs:
        run_id = job["run_id"]
        mode = job["mode"]
        test_csv = Path(job["test_csv"])
        emb_path = Path(job["embed_test_path"])

        print(f"\n[INFO] evaluating ({mode}) run_id={run_id}")
        print(f"[INFO] test_csv={test_csv}")
        print(f"[INFO] emb_path={emb_path}")

        # --- load embedding / labels ---
        X, ips = load_ip_embeddings(emb_path)
        y = make_ip_labels(test_csv, ips)

        n_ip = len(y)
        n_attack = int(y.sum())
        print(f"[INFO] #IP={n_ip}, #attack={n_attack}")

        # --- scores ---
        proto_scores = proto_loocv_scores(X, y)
        proto_auc, proto_ap = safe_auc_ap(y, proto_scores)
        proto_r = recall_at_num_attack(y, proto_scores)

        knn_metrics = {}
        knn_scores_for_table: Optional[np.ndarray] = None
        for k in KNN_K_LIST:
            s = knn_loocv_scores(X, y, k=k)
            auc, ap = safe_auc_ap(y, s)
            r = recall_at_num_attack(y, s)
            knn_metrics[f"knn{k}_auc"] = auc
            knn_metrics[f"knn{k}_ap"] = ap
            knn_metrics[f"knn{k}_recall_at_num_attack"] = r

            if k == KNN_IP_TABLE_K:
                knn_scores_for_table = s

        # --- run-level record ---
        rec = {
            "mode": mode,
            "dataset": DATASET,
            "run_id": run_id,
            "n_ip": n_ip,
            "n_attack": n_attack,
            "proto_auc": proto_auc,
            "proto_ap": proto_ap,
            "proto_recall_at_num_attack": proto_r,
            **knn_metrics,
        }
        run_records.append(rec)

        # --- IP-level tables (proto + one kNN) ---
        df_ip_proto = summarize_ip_results(ips, y, proto_scores, score_name="proto_score")
        ip_csv_proto = out_dir / f"ip_table_proto_{run_id}.csv"
        df_ip_proto.to_csv(ip_csv_proto, index=False)
        ip_records_paths.append(str(ip_csv_proto))

        if knn_scores_for_table is None:
            # if user did not include KNN_IP_TABLE_K in list, fallback to first k
            k0 = KNN_K_LIST[0]
            knn_scores_for_table = knn_loocv_scores(X, y, k=k0)
            knn_name = f"knn{k0}_score"
        else:
            knn_name = f"knn{KNN_IP_TABLE_K}_score"

        df_ip_knn = summarize_ip_results(ips, y, knn_scores_for_table, score_name=knn_name)
        ip_csv_knn = out_dir / f"ip_table_{knn_name}_{run_id}.csv"
        df_ip_knn.to_csv(ip_csv_knn, index=False)
        ip_records_paths.append(str(ip_csv_knn))

        # quick console peek: show failures
        missed_attack = df_ip_knn[(df_ip_knn["true_label"] == 1) & (~df_ip_knn["success"])]
        if len(missed_attack) > 0:
            print("[INFO] missed attack IPs (by rank criterion):")
            print(missed_attack[["ip", "rank", knn_name]].to_string(index=False))
        else:
            print("[INFO] no missed attack IPs (by rank criterion).")

    # --- save run-level summary ---
    df_out = pd.DataFrame(run_records)
    out_csv = out_dir / "ip_embedding_eval_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print("\n[INFO] saved run summary:", out_csv)

    # --- optional: also write list of per-IP tables created ---
    out_list = out_dir / "ip_tables_created.txt"
    out_list.write_text("\n".join(ip_records_paths) + "\n")
    print("[INFO] saved ip table list:", out_list)


if __name__ == "__main__":
    main()
