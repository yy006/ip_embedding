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
EXPERIMENT = '2025-12-02T12-53-23_incremental_zr9cmvxb'
#EXPERIMENT = '2025-11-21T06-50-31_incremental_xttsbif7'
#EXPERIMENT = '2025-09-30T05-54-05_single_4vfhlp7f'
json_path = f'experiments/{DATASET}/{EXPERIMENT}/experiment.json'

# ========= ここだけ編集してください =========
rand8 = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=8))
#INPUT_CSV      = "datasets/UNSW-NB15/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h/2015021802_2015021804_by2h.csv"
OUT_DIR        = f"eval/eval_anomaly_{EXPERIMENT}_{rand8}"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

with open(json_path, 'r') as f:
    config = json.load(f)

# テストデータ
INPUT_CSV      = config['blocks']['6']

# 埋め込みのパス
#EMBED_PKL_TRAIN = config['results']['blocks']['005']['model']['model_path']
#EMBED_PKL_TEST  = config['results']['blocks']['005']['model']['model_path']
#EMBED_PKL_TRAIN = "/workspace/experiments/UNSW-NB15/2025-11-28T08-49-24_single_hoy9fcim/models/model_block_001"   
#EMBED_PKL_TEST  = "/workspace/experiments/UNSW-NB15/2025-11-28T08-48-35_single_iwfuomb2/models/model_block_001" 
EMBED_PKL_TRAIN = "/workspace/experiments/"+ DATASET + "/" + EXPERIMENT + "/models/model_block_005"
EMBED_PKL_TEST  = "/workspace/experiments/"+ DATASET + "/" + EXPERIMENT + "/models/model_block_006"

# 埋め込みの読み込み
def load_embeddings(path: str | Path):
    p = Path(path)
    obj = torch.load(p, map_location="cpu", weights_only=False)

    print (obj)
    return obj

model_train = load_embeddings(EMBED_PKL_TRAIN)
model_test  = load_embeddings(EMBED_PKL_TEST)

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
    """
    gensim互換（model.wv, get_vector が使える）を前提に安全にアクセスするためのラッパを返す。
    """
    # gensim >= 4 なら model.wv.key_to_index / model.wv.vectors / get_vector が使える
    wv = getattr(model_obj, "wv", None)
    if wv is None:
        # まれに KeyedVectors をそのままpickleしている場合もある
        wv = model_obj
    # 確認
    if not hasattr(wv, "key_to_index") or not hasattr(wv, "vectors") or not hasattr(wv, "get_vector"):
        raise TypeError("埋め込みモデルが gensim KeyedVectors 互換ではありません。model.wv.get_vector が必要です。")
    return wv

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

# ========= データ読み込み & 前処理 =========
OUT_DIR = ensure_outdir(OUT_DIR)
df = load_csv(INPUT_CSV)

# 'Label' 列チェック
if "Label" not in df.columns:
    raise KeyError("Label 列が見つかりません。列名を確認してください。")

# 埋め込みインターフェース
wv_train = get_embedding_interface(model_train)
wv_test  = get_embedding_interface(model_test)
mean_vec_train = compute_mean_vector(wv_train)
mean_vec_test  = compute_mean_vector(wv_test)

# 使う列（存在確認）— まずは“元特徴のみ”で判定（埋め込みは後で付与）
USE_COLS = select_existing_columns(df, USE_COLS)
# カテゴリ/数値の自動判定（元特徴のみ）
CAT_COLS = [c for c in ["proto", "state", "service"] if c in USE_COLS]
NUM_COLS = [c for c in USE_COLS if c not in CAT_COLS]

# 学習/評価用に必要なカラム（埋め込み付与用に srcip を残す）
need_cols = ["Label", "srcip"] + CAT_COLS + NUM_COLS
work = df[need_cols].copy()

X = work.drop(columns=["Label"])
y = work["Label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
# ========= 埋め込みを train/test で別モデルから付与 =========
# （emb_cols の列名は train 側の次元に合わせて共通化）
X_train, emb_cols = attach_srcip_embedding(X_train.copy(), wv_train, mean_vec_train, ip_col="srcip")
X_test,  _        = attach_srcip_embedding(X_test.copy(),  wv_test,  mean_vec_test,  ip_col="srcip")

# 学習に srcip 本体は使わない（列指定に含めない）
ALL_NUM_COLS = NUM_COLS + emb_cols

# ========= 変換パイプライン（OneHotEncoderはdense出力にしてHGBに対応） =========
cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
num_transformer = "passthrough"  # 標準化は木系では必須でないので省略
preprocess = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, CAT_COLS),
        ("num", num_transformer, ALL_NUM_COLS),
    ],
    remainder="drop",
)

# ========= 1) 正常のみで学習する異常検知（IsolationForest） =========
# 訓練は Benign のみ
mask_benign_train = (y_train == BENIGN_LABEL)
X_train_benign = X_train[mask_benign_train]

iso_clf = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination="auto",  # 汎用設定（テストの異常率は使わない）
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipe_iso = Pipeline([
    ("prep", preprocess),
    ("clf", iso_clf),
])

pipe_iso.fit(X_train_benign)  # 異常を含めない学習

# スコア：decision_function は「+ = inlier（正常）, - = outlier（異常）」なので符号を反転して異常スコア化
dec = pipe_iso.decision_function(X_test)  # 高いほど正常
anom_score = -dec                        # 高いほど異常

# ROC-AUC / AP（Label=1を異常とみなす）
try:
    roc_if = roc_auc_score(y_test, anom_score)
except ValueError:
    roc_if = None
try:
    ap_if = average_precision_score(y_test, anom_score)
except ValueError:
    ap_if = None

# 閾値は 0（= decision_function<0.03 を異常）相当
y_pred_if = (anom_score > 0.03).astype(int)
cm_if = confusion_matrix(y_test, y_pred_if).tolist()

rep_if = RunReport(
    setting="one_class_isolation_forest",
    n_train=int(mask_benign_train.sum()),
    n_test=len(y_test),
    roc_auc=roc_if,
    ap=ap_if,
    threshold_desc="decision_function<0 を異常（異常スコア>0）として判定",
    confusion=cm_if,
    notes=f"CAT={CAT_COLS}, NUM={NUM_COLS}, EMBED_DIM={len(emb_cols)}"
)
save_report_json(rep_if, OUT_DIR, "report_isoforest.json")

# 予測詳細を保存
pred_if = pd.DataFrame({
    "y_true": y_test.values,
    "anom_score": anom_score,
    "y_pred": y_pred_if,
})
pred_if.to_csv(Path(OUT_DIR) / "pred_isoforest.csv", index=False)

print("[IsolationForest] ROC-AUC:", roc_if, " AP:", ap_if)
print("[IsolationForest] Confusion matrix:\n", np.array(cm_if))

# ========= 2) 正常+攻撃の両方で学習する教師あり分類（HistGradientBoosting） =========
hgb = HistGradientBoostingClassifier(
    max_depth=None,
    learning_rate=0.1,
    max_iter=300,
    random_state=RANDOM_STATE,
)

pipe_hgb = Pipeline([
    ("prep", preprocess),
    ("clf", hgb),
])

pipe_hgb.fit(X_train, y_train)

# 確率（正例=ATTACK_LABEL の確率）を取りたいので predict_proba 相当を取得
# HGBClassifier は predict_proba を提供（binary の場合）
if hasattr(pipe_hgb.named_steps["clf"], "predict_proba"):
    prob = pipe_hgb.predict_proba(X_test)[:, 1]
else:
    # ない場合は decision_function をシグモイドで近似…だが、HGB は基本 prob あり
    # 念のため fallback
    raw = pipe_hgb.decision_function(X_test)
    prob = 1.0 / (1.0 + np.exp(-raw))

# ROC-AUC / AP
try:
    roc_sup = roc_auc_score(y_test, prob)
except ValueError:
    roc_sup = None
try:
    ap_sup = average_precision_score(y_test, prob)
except ValueError:
    ap_sup = None

# 閾値は0.5
y_pred_sup = (prob >= 0.5).astype(int)
cm_sup = confusion_matrix(y_test, y_pred_sup).tolist()

rep_sup = RunReport(
    setting="supervised_hgb_classifier",
    n_train=len(y_train),
    n_test=len(y_test),
    roc_auc=roc_sup,
    ap=ap_sup,
    threshold_desc="P(Attack) >= 0.5 を異常と判定",
    confusion=cm_sup,
    notes=f"CAT={CAT_COLS}, NUM={NUM_COLS}, EMBED_DIM={len(emb_cols)}"
)
save_report_json(rep_sup, OUT_DIR, "report_supervised_hgb.json")

pred_sup = pd.DataFrame({
    "y_true": y_test.values,
    "prob_attack": prob,
    "y_pred": y_pred_sup,
})
pred_sup.to_csv(Path(OUT_DIR) / "pred_supervised_hgb.csv", index=False)

print("[Supervised HGB] ROC-AUC:", roc_sup, " AP:", ap_sup)
print("[Supervised HGB] Confusion matrix:\n", np.array(cm_sup))

print(f"[DONE] 結果を {OUT_DIR} に保存しました。レポート: report_isoforest.json / report_supervised_hgb.json")

# ========= 追加：ROC曲線をCSV/PNGで保存 =========
# IF は anom_score（大きいほど異常=Attack）を渡す
save_and_print_roc(y_test.values, anom_score, OUT_DIR, prefix="isoforest")
# 監督ありは Attack の確率を渡す
save_and_print_roc(y_test.values, prob, OUT_DIR, prefix="supervised_hgb")

# ========= 追加：異常スコアの“件数”ヒストグラム（横軸=スコア）を保存 =========
# IF: しきい値は decision_function<0 → anom_score>0
save_score_count_hist(y_test.values, anom_score, OUT_DIR, prefix="isoforest",      bins=60, thr=0.0)
# supervised: しきい値は 0.5
save_score_count_hist(y_test.values, prob,        OUT_DIR, prefix="supervised_hgb", bins=60, thr=0.5)

print("y_test の件数と内訳:", y_test.value_counts())
print("X_test のサイズ:", X_test.shape)

print("異常スコア（anom_score）の一意な値:", np.unique(anom_score))
print("最大・最小:", anom_score.min(), anom_score.max())

print("y_test の件数と内訳:\n", y_test.value_counts())



# ========= 追加：HGB の特徴量重要度（モデル内 + Permutation）を保存 =========
def _get_output_feature_names(prep, input_cols):
    try:
        return prep.get_feature_names_out(input_cols)
    except Exception:
        names = []
        for name, trans, cols in prep.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                base = trans.get_feature_names_out(cols)
                names.extend([f"{name}__{b}" for b in base])
            else:
                names.extend([f"{name}__{c}" for c in cols])
        return np.array(names, dtype=object)

def _save_importances(names, values, out_dir: Path, prefix: str, top_k: int = 30, title: str = ""):
    order = np.argsort(values)[::-1]
    names_top = np.array(names)[order][:top_k]
    vals_top  = np.array(values)[order][:top_k]
    # CSV
    imp_df = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    imp_df.to_csv(Path(out_dir) / f"feature_importance_{prefix}.csv", index=False)
    # 図
    plt.figure(figsize=(8, max(4, 0.3*len(names_top))))
    plt.barh(range(len(names_top)), vals_top[::-1])
    plt.yticks(range(len(names_top)), names_top[::-1], fontsize=9)
    plt.xlabel("Importance")
    plt.title(title or f"Feature importance - {prefix}")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"feature_importance_{prefix}.png", dpi=180, bbox_inches="tight")
    plt.close()

# 展開後の特徴量名
_prep = pipe_hgb.named_steps["prep"]
_clf  = pipe_hgb.named_steps["clf"]
_feat_names = _get_output_feature_names(_prep, X.columns)

# 1) モデル内（不純度）重要度
if hasattr(_clf, "feature_importances_"):
    _save_importances(_feat_names, _clf.feature_importances_, OUT_DIR,
                      prefix="supervised_hgb_model", title="HGB impurity-based importance")

#2) Permutation Importance（高速化版）
#    - 生の入力列（OneHot前）単位で評価 → 列数が少なく高速
#    - 評価データを最大5000件にサブサンプル
_n_sub = min(5000, len(X_test))
_X_sub = X_test.sample(n=_n_sub, random_state=RANDOM_STATE)
_y_sub = y_test.loc[_X_sub.index]
_pi = permutation_importance(
    pipe_hgb,                # パイプライン全体に対して
    _X_sub, _y_sub,
    scoring="roc_auc",       # FutureWarning回避
    n_repeats=3,             # 軽量化（必要に応じて増やす）
    random_state=RANDOM_STATE,
    n_jobs=-1
)
# 生列名で保存
_save_importances(np.array(X.columns), _pi.importances_mean, OUT_DIR,
                  prefix="supervised_hgb_perm",
                  title="HGB permutation importance (ROC-AUC, raw features, subsampled)")

from sklearn.metrics import roc_auc_score
s = pd.Series(prob, index=X_test.index)  # 既にある予測確率

def single_feature_auc(col):
    """
    HGBの前処理(prep)でフルデータを変換→対象列に対応する展開後の列だけ使って
    軽い分類器(LogReg)を学習し、単一特徴のAUCを返す。
    ※ pipe_hgb.fit(...) 実行後に呼んでください。
    """
    prep = pipe_hgb.named_steps["prep"]
    # フルを一度だけ変換
    Xt_tr = prep.transform(X_train)
    Xt_te = prep.transform(X_test)
    # 展開後の列名を取得
    try:
        out_names = prep.get_feature_names_out(X.columns)
    except Exception:
        # 古いsklearn用フォールバック
        out_names = []
        for name, trans, cols in prep.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                base = trans.get_feature_names_out(cols)
                out_names.extend([f"{name}__{b}" for b in base])
            else:
                out_names.extend([f"{name}__{c}" for c in cols])
        out_names = np.array(out_names, dtype=object)
    # 対象列に対応する展開後インデックス（OneHotは複数列）
    if col in CAT_COLS:
        prefix = f"cat__{col}_"
        mask = np.array([n.startswith(prefix) for n in out_names])
    else:
        prefix = f"num__{col}"
        mask = (out_names == prefix)
    idx = np.where(mask)[0]
    if idx.size == 0:
        print(f"[WARN] no expanded features for {col}")
        return np.nan
    # 単一特徴（展開後の該当列群）のみで小さなモデルを学習
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xt_tr[:, idx], y_train)
    prob = clf.predict_proba(Xt_te[:, idx])[:, 1]
    return roc_auc_score(y_test, prob)

for col in ["src_emb_0","src_emb_28"]:
    if col in X_test.columns:
        print(col, single_feature_auc(col))