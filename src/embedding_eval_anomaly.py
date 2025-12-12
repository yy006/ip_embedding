import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, roc_curve, make_scorer
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

# 実験設定の読み込み
DATASET = 'UNSW-NB15'
EXPERIMENT = '2025-12-05T04-36-49_incremental_1a9temld'
#EXPERIMENT = '2025-11-28T10-25-38_incremental_ohosptwi'
#EXPERIMENT = '2025-09-30T05-54-05_single_4vfhlp7f'
json_path = f'experiments/{DATASET}/{EXPERIMENT}/experiment.json'

# ========= ここだけ編集してください =========
rand8 = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=8))
#INPUT_CSV      = "datasets/UNSW-NB15/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h/2015021802_2015021804_by2h.csv"
OUT_DIR        = f"eval/eval_anomaly_{EXPERIMENT}_{rand8}"
TEST_SIZE      = 0.2
RANDOM_STATE   = 46
with open(json_path, 'r') as f:
    config = json.load(f)

# テストデータ
INPUT_CSV      = config['blocks']['6']

# 埋め込みのパス
#EMBED_PKL_TRAIN = config['results']['blocks']['005']['model']['model_path']
#EMBED_PKL_TEST  = config['results']['blocks']['005']['model']['model_path']
#EMBED_PKL_TRAIN = "/workspace/experiments/UNSW-NB15/2025-12-05T09-00-27_single_mwjnnrym/models/model_block_001"   
#EMBED_PKL_TEST  = "/workspace/experiments/UNSW-NB15/2025-12-05T06-39-23_single_cbzfz5go/models/model_block_001" 
EMBED_PKL_TRAIN = "/workspace/experiments/"+ DATASET + "/" + EXPERIMENT + "/models/model_block_005"
EMBED_PKL_TEST  = "/workspace/experiments/"+ DATASET + "/" + EXPERIMENT + "/models/model_block_006"

# 埋め込みの読み込み
def load_embeddings(path: str | Path):
    p = Path(path) 
    '''
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj
    '''
    return torch.load(p, map_location="cpu", weights_only=False)

model_train = load_embeddings(EMBED_PKL_TRAIN)
model_test  = load_embeddings(EMBED_PKL_TEST)

print("model_test:", model_test)

# 使う列 sttl抜き
USE_COLS = ["proto","state","dur", "sbytes","dbytes","sloss","dloss","service","Sload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","trans_depth","res_bdy_len","Sjit","Djit","sttl"]
#USE_COLS =[]
# ==========================================

# ========= ここから追記 =========
import os
from functools import partial
from dataclasses import dataclass, asdict

# --- ラベル定義（※ 1=Attack, 0=Benign で実装。異なる場合はこの2行だけ変更してください） ---
ATTACK_LABEL = 1
BENIGN_LABEL = 0

# ========= 共通ユーティリティ =========
def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

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

def compute_mean_vector(wv) -> np.ndarray:
    if getattr(wv, "vectors", None) is None or len(wv.vectors) == 0:
        raise ValueError("埋め込みベクトルが空です。学習済みモデルを確認してください。")
    return wv.vectors.mean(axis=0)

def ip_to_vec(ip: str, wv, mean_vec: np.ndarray) -> np.ndarray:
    try:
        return wv.get_vector(ip)
    except KeyError:
        return mean_vec  # 語彙外は平均ベクトルで埋める（要望どおり）

def attach_srcip_embedding(df: pd.DataFrame, wv, mean_vec: np.ndarray, ip_col: str = "srcip") -> pd.DataFrame:
    if ip_col not in df.columns:
        raise KeyError(f"{ip_col} 列が見つかりません。現在の列: {list(df.columns)[:20]} ...")
    dim = wv.vectors.shape[1]
    # ベクトル化
    vecs = np.vstack([ip_to_vec(ip, wv, mean_vec) for ip in df[ip_col].astype(str).values])
    # 列として結合
    emb_cols = [f"src_emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(vecs, columns=emb_cols, index=df.index)
    return pd.concat([df, emb_df], axis=1), emb_cols

def select_existing_columns(df: pd.DataFrame, want: List[str]) -> List[str]:
    exist = [c for c in want if c in df.columns]
    missing = [c for c in want if c not in df.columns]
    if missing:
        print(f"[WARN] 入力CSVに存在しない列をスキップします: {missing}")
    return exist

def save_and_print_roc(y_true, score, out_dir: Path, prefix: str):
     """
     y_true: 0/1（1=Attack=positive）
     score : 大きいほど Attack である確率・スコア
     """
     try:
         fpr, tpr, thr = roc_curve(y_true, score, pos_label=1)
         auc = roc_auc_score(y_true, score)
     except ValueError as e:
         print(f"[{prefix}] ROCを計算できません: {e}")
         return
     # 1) コンソールに主要点を出す（頭2行＋末尾1行）
     print(f"[{prefix}] ROC-AUC = {auc:.6f}")
     print(f"[{prefix}] ROC head:")
     for i in range(min(2, len(fpr))):
         print(f"  thr={thr[i]:.6f}  fpr={fpr[i]:.6f}  tpr={tpr[i]:.6f}")
     if len(fpr) > 0:
         print(f"  ... last: thr={thr[-1]:.6f}  fpr={fpr[-1]:.6f}  tpr={tpr[-1]:.6f}")
     # 2) CSV 保存
     roc_df = pd.DataFrame({"threshold": thr, "fpr": fpr, "tpr": tpr})
     roc_csv = out_dir / f"roc_{prefix}.csv"
     roc_df.to_csv(roc_csv, index=False)
     # 3) 図保存
     plt.figure()
     plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
     plt.plot([0,1], [0,1], linestyle="--")
     plt.xlabel("False Positive Rate")
     plt.ylabel("True Positive Rate")
     plt.title(f"ROC - {prefix}")
     plt.legend(loc="lower right")
     roc_png = out_dir / f"roc_{prefix}.png"
     plt.savefig(roc_png, dpi=180, bbox_inches="tight")
     plt.close()
     print(f"[{prefix}] ROC CSV: {roc_csv}")
     print(f"[{prefix}] ROC PNG: {roc_png}")

def save_score_count_hist(y_true, score, out_dir: Path, prefix: str, *, bins: int = 50, thr: float | None = None):
     """
     異常スコア（または Attack 確率）を横軸とし、Attack/Benign の“件数”を同一ビンでカウントして
     積み上げ棒グラフにして保存。あわせてCSV（各ビンの左端/右端と件数）を出力。
     """
     y_true = np.asarray(y_true)
     score  = np.asarray(score)
     if score.size == 0:
         print(f"[{prefix}] empty score array; skip count hist.")
         return
     s_min, s_max = float(np.min(score)), float(np.max(score))
     if s_min == s_max:
         # スコアが全て同じ場合は±1e-6だけ広げる
         s_min -= 1e-6
         s_max += 1e-6
     edges = np.linspace(s_min, s_max, bins + 1)
     # 同じビン境界でBenign/Attackをカウント
     cnt_benign, _ = np.histogram(score[y_true == 0], bins=edges)
     cnt_attack, _ = np.histogram(score[y_true == 1], bins=edges)
     mids = (edges[:-1] + edges[1:]) / 2.0
     width = edges[1] - edges[0]
     # CSV保存
     out_csv = Path(out_dir) / f"score_count_hist_{prefix}.csv"
     pd.DataFrame({
         "bin_left": edges[:-1],
         "bin_right": edges[1:],
         "midpoint": mids,
         "count_benign": cnt_benign,
         "count_attack": cnt_attack,
         "count_total": cnt_benign + cnt_attack,
     }).to_csv(out_csv, index=False)
     # プロット（積み上げバー）
     plt.figure(figsize=(10, 4))
     plt.bar(edges[:-1], cnt_benign, align="edge", width=width, alpha=0.75, label="Benign (count)")
     plt.bar(edges[:-1], cnt_attack, align="edge", width=width, alpha=0.75, bottom=cnt_benign, label="Attack (count)")
     if thr is not None:
         plt.axvline(thr, linestyle="--", linewidth=1.2, label=f"threshold={thr}")
     plt.xlabel("Anomaly score / P(Attack)")
     plt.ylabel("Count")
     plt.title(f"Score count histogram - {prefix}")
     plt.legend()
     out_png = Path(out_dir) / f"score_count_hist_{prefix}.png"
     plt.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close()
     print(f"[{prefix}] Score count hist CSV:", out_csv)
     print(f"[{prefix}] Score count hist PNG:", out_png)

@dataclass
class RunReport:
    setting: str
    n_train: int
    n_test: int
    roc_auc: float | None
    ap: float | None  # average precision (PR-AUC 近似)
    threshold_desc: str
    confusion: List[List[int]] | None  # [[tn, fp],[fn, tp]] ではなく sklearn の表示順に合わせます
    notes: str = ""

def save_report_json(report: RunReport, out_dir: Path, fname: str):
    with open(out_dir / fname, "w") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

# ========= データ読み込み & 前処理（ここは1回だけ実行） =========
OUT_DIR = ensure_outdir(OUT_DIR)
df = load_csv(INPUT_CSV)

if "Label" not in df.columns:
    raise KeyError("Label 列が見つかりません。列名を確認してください。")

# 埋め込みインターフェース
wv_train = get_embedding_interface(model_train)
wv_test  = get_embedding_interface(model_test)
mean_vec_train = compute_mean_vector(wv_train)
mean_vec_test  = compute_mean_vector(wv_test)

# 使う列（存在確認）
USE_COLS = select_existing_columns(df, USE_COLS)
CAT_COLS = [c for c in ["proto", "state", "service"] if c in USE_COLS]
NUM_COLS = [c for c in USE_COLS if c not in CAT_COLS]


def run_isoforest_for_seed(seed: int) -> dict:
    """
    1つの seed について IF を学習・評価し、
    AUC などのメトリクスと保存先を dict で返す。
    出力は OUT_DIR/seed_{seed}/ 以下にまとめる。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir_seed = ensure_outdir(Path(OUT_DIR) / f"seed_{seed}")

    # --- 特徴量とラベルを作成 ---
    need_cols = ["Label", "srcip"] + CAT_COLS + NUM_COLS
    work = df[need_cols].copy()

    X = work.drop(columns=["Label"])
    y = work["Label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    # --- 埋め込み付与（train/test で別モデル） ---
    X_train, emb_cols = attach_srcip_embedding(
        X_train.copy(), wv_train, mean_vec_train, ip_col="srcip"
    )
    X_test, _ = attach_srcip_embedding(
        X_test.copy(), wv_test, mean_vec_test, ip_col="srcip"
    )

    ALL_NUM_COLS = NUM_COLS + emb_cols

    # 前処理パイプライン（OneHot + 数値＋埋め込み）
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_transformer = "passthrough"

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CAT_COLS),
            ("num", num_transformer, ALL_NUM_COLS),
        ],
        remainder="drop",
    )

    # --- Benign のみで IF 学習 ---
    mask_benign_train = (y_train == BENIGN_LABEL)
    X_train_benign = X_train[mask_benign_train]

    iso_clf = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )

    pipe_iso = Pipeline([
        ("prep", preprocess),
        ("clf", iso_clf),
    ])

    pipe_iso.fit(X_train_benign)

    # --- スコア & メトリクス ---
    dec = pipe_iso.decision_function(X_test)  # 高いほど正常
    anom_score = -dec                         # 高いほど異常

    try:
        roc_if = roc_auc_score(y_test, anom_score)
    except ValueError:
        roc_if = None

    try:
        ap_if = average_precision_score(y_test, anom_score)
    except ValueError:
        ap_if = None

    # しきい値は例として 0.0（= decision_function < 0 を異常）
    y_pred_if = (anom_score > 0.0).astype(int)
    cm_if = confusion_matrix(y_test, y_pred_if).tolist()

    rep_if = RunReport(
        setting=f"one_class_isolation_forest_seed_{seed}",
        n_train=int(mask_benign_train.sum()),
        n_test=len(y_test),
        roc_auc=roc_if,
        ap=ap_if,
        threshold_desc="decision_function<0 を異常（異常スコア>0）として判定",
        confusion=cm_if,
        notes=f"seed={seed}, CAT={CAT_COLS}, NUM={NUM_COLS}, EMBED_DIM={len(emb_cols)}",
    )
    save_report_json(rep_if, out_dir_seed, "report_isoforest.json")

    # 予測詳細
    pred_if = pd.DataFrame({
        "y_true": y_test.values,
        "anom_score": anom_score,
        "y_pred": y_pred_if,
    })
    pred_if.to_csv(out_dir_seed / "pred_isoforest.csv", index=False)

    print(f"[seed={seed}] ROC-AUC:", roc_if, " AP:", ap_if)
    print(f"[seed={seed}] Confusion matrix:\n", np.array(cm_if))

    # ROC曲線＆ヒストグラムも seed ごとに保存
    save_and_print_roc(y_test.values, anom_score, out_dir_seed, prefix=f"isoforest_seed{seed}")
    save_score_count_hist(y_test.values, anom_score, out_dir_seed,
                          prefix=f"isoforest_seed{seed}", bins=60, thr=0.0)

    return {
        "seed": seed,
        "n_train": int(mask_benign_train.sum()),
        "n_test": len(y_test),
        "roc_auc": roc_if,
        "ap": ap_if,
    }

if __name__ == "__main__":
    # 回したい seed の範囲（10〜20）
    seeds = list(range(40, 51))

    results = []
    for s in seeds:
        print(f"\n===== run seed={s} =====")
        res = run_isoforest_for_seed(s)
        results.append(res)

    # DataFrame にして per-seed 結果を CSV で保存
    results_df = pd.DataFrame(results)
    results_csv = Path(OUT_DIR) / "isoforest_seed_results.csv"
    results_df.to_csv(results_csv, index=False)
    print("Per-seed results CSV:", results_csv)

    # AUC の統計量を計算（None を除外）
    aucs = [r["roc_auc"] for r in results if r["roc_auc"] is not None]

    if aucs:
        auc_mean = float(np.mean(aucs))
        auc_max  = float(np.max(aucs))
        auc_min  = float(np.min(aucs))
        auc_med  = float(np.median(aucs))
        auc_std  = float(np.std(aucs))
    else:
        auc_mean = auc_max = auc_min = auc_med = auc_std = None

    # まとめを JSON で保存
    summary = {
        "seeds": seeds,
        "n_runs": len(aucs),
        "roc_auc_mean": auc_mean,
        "roc_auc_max":  auc_max,
        "roc_auc_min":  auc_min,
        "roc_auc_median": auc_med,
        "roc_auc_std": auc_std,
    }

    summary_json = Path(OUT_DIR) / "isoforest_seed_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("AUC summary JSON:", summary_json)
    print("AUC stats:",
          "mean=", auc_mean,
          "max=", auc_max,
          "min=", auc_min,
          "median=", auc_med,
          "std=", auc_std)

