import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional


from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix
)
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest


# 実験設定の読み込み
DATASET = 'UNSW-NB15'
EXPERIMENT = '2025-10-07T10-48-21_incremental_g9yil884'
#EXPERIMENT = '2025-10-07T01-57-11_incremental_jk7mc49n'
#EXPERIMENT = '2025-09-30T05-54-05_single_4vfhlp7f'
json_path = f'experiments/{DATASET}/{EXPERIMENT}/experiment.json'

# ========= ここだけ編集してください =========
INPUT_CSV      = "datasets/UNSW-NB15/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h/2015021802_2015021804_by2h.csv"
OUT_DIR        = f"{EXPERIMENT}_003_out_run"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

with open(json_path, 'r') as f:
    config = json.load(f)

# 埋め込みのパス
EMBED_PKL = config['results']['blocks']['003']['model']['model_path']

# 埋め込みの読み込み
def load_embeddings(path: str | Path):
    p = Path(path)
    with open(p, "rb") as f:
        obj = pickle.load(f)

    print (obj)
    return obj

model = load_embeddings(EMBED_PKL)

# 使う列
USE_COLS = ["proto","state","dur", "sbytes","dbytes","sttl","dttl","sloss","dloss","service","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len","Sjit","Djit"]
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
wv = get_embedding_interface(model)
mean_vec = compute_mean_vector(wv)

# 埋め込み付与（srcip のみ）
df, emb_cols = attach_srcip_embedding(df, wv, mean_vec, ip_col="srcip")

# 使う列（存在確認）
USE_COLS = select_existing_columns(df, USE_COLS)
# カテゴリ/数値の自動判定（与えられた想定通り）
CAT_COLS = [c for c in ["proto", "state", "service"] if c in USE_COLS]
NUM_COLS = [c for c in USE_COLS if c not in CAT_COLS]
# 埋め込み列は数値として常に使用
ALL_NUM_COLS = NUM_COLS + emb_cols

# 学習/評価用に必要なカラムだけサブセット
need_cols = ["Label"] + CAT_COLS + ALL_NUM_COLS
work = df[need_cols].copy()

# ========= 変換パイプライン（OneHotEncoderはdense出力にしてHGBに対応） =========
cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
num_transformer = "passthrough"  # 標準化は木系では必須でないので省略（必要なら StandardScaler に変更可）

preprocess = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, CAT_COLS),
        ("num", num_transformer, ALL_NUM_COLS),
    ],
    remainder="drop",
)

# ========= スプリット =========
X = work.drop(columns=["Label"])
y = work["Label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
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

# 閾値は 0（= decision_function<0 を異常）相当
y_pred_if = (anom_score > 0).astype(int)
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