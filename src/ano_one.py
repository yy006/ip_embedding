import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, roc_curve, make_scorer
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier


# 実験設定の読み込み
DATASET = 'UNSW-NB15'
EXPERIMENT = '2025-10-07T19-47-12_incremental_8li7mu1w'
#EXPERIMENT = '2025-10-07T01-57-11_incremental_jk7mc49n'
#EXPERIMENT = '2025-09-30T05-54-05_single_4vfhlp7f'
json_path = f'experiments/{DATASET}/{EXPERIMENT}/experiment.json'

# ========= ここだけ編集してください =========
INPUT_CSV      = "datasets/UNSW-NB15/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h/2015021802_2015021804_by2h.csv"
OUT_DIR        = f"{EXPERIMENT}_005_out_run"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

with open(json_path, 'r') as f:
    config = json.load(f)

# 埋め込みのパス
embed_path = config['results']['blocks']['005']['model']['model_path']

# 埋め込みの読み込み
def load_embeddings(path: str | Path):
    p = Path(path)
    with open(p, "rb") as f:
        obj = pickle.load(f)

    print (obj)
    return obj

model = load_embeddings(embed_path)

USE_COLS =[]

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

#df = load_csv(INPUT_CSV)

#training_df = load_csv("datasets/UNSW-NB15/Training and Testing Sets/UNSW_NB15_training-set.csv")
training_df = load_csv("datasets/UNSW-NB15/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h/2015012800_2015012802_by2h.csv")
#testing_df = load_csv("datasets/UNSW-NB15/Training and Testing Sets/UNSW_NB15_testing-set.csv")
testing_df = load_csv("datasets/UNSW-NB15/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h/2015021802_2015021804_by2h.csv")

training_df = training_df[["Label", "sttl", "sbytes"]]
testing_df = testing_df[["Label", "sttl", "sbytes"]]

# ========= スプリット =========
#X = df.drop(columns=["Label"])
#y = df["Label"].astype(int)

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
#)
X_train = training_df.drop(columns=["Label"])
y_train = training_df["Label"].astype(int)

X_test  = testing_df.drop(columns=["Label"])
y_test  = testing_df["Label"].astype(int)

# --- 前処理の列は X_train から決める（← NameError対策） ---
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# ========= 2) 正常+攻撃の両方で学習する教師あり分類（HistGradientBoosting） =========
#hgb = HistGradientBoostingClassifier(
#    max_depth=None,
#    learning_rate=0.1,
#    max_iter=300,
#    random_state=RANDOM_STATE,
#)

rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=1,
        random_state=RANDOM_STATE,
        class_weight=None,   # 少数クラス重み付けしたいなら "balanced"
    )

# Define preprocessing step
#preprocess = ColumnTransformer(
#    transformers=[
#        ("num", StandardScaler(), X.select_dtypes(include=np.number).columns.tolist()),
##        ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include="object").columns.tolist()),
  #  ]
#)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", rf),
])

pipe.fit(X_train, y_train)

# 確率（正例=ATTACK_LABEL の確率）を取りたいので predict_proba 相当を取得
# HGBClassifier は predict_proba を提供（binary の場合）
if hasattr(pipe.named_steps["clf"], "predict_proba"):
    prob = pipe.predict_proba(X_test)[:, 1]

roc_sup = roc_auc_score(y_test, prob)
ap_sup = average_precision_score(y_test, prob)


# 閾値は0.5
y_pred_sup = (prob >= 0.5).astype(int)
cm_sup = confusion_matrix(y_test, y_pred_sup).tolist()

# 結果の表示
print("=== Supervised (RF) ===")
print(f"ROC-AUC: {roc_sup:.4f}")
print(f"AP: {ap_sup:.4f}")
print("Confusion Matrix (threshold=0.5):")
print(cm_sup)
print()
# 結果の保存
