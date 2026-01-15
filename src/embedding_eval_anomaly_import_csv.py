import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

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

import os
from dataclasses import dataclass, asdict

# ============================================================
# 設定エリア（ここだけいじればOK）
# ============================================================

# --- 手書きモード / CSVモード 切り替え ---
USE_RUNS_CSV = True  # False: 手書きEXPERIMENTで1本評価 / True: CSVのincremental run_id群を回す
RUNS_CSV_PATH = ARTIFACTS_ROOT / "alpha_sweep_mapping_a82g3rke_None.csv"  # CSVモード時に読むファイル

#OUT_DIR_NAME = f"eval_anomaly_{DATASET}"
OUT_DIR_NAME = "incre_dup_true_IFdataimprove"

# single モード用のCSVファイルパス（RUNS_CSV_PATH と別に指定）
mode_single = False
# train側path
SINGLE_RUNS_CSV_A = ARTIFACTS_ROOT / "alpha_sweep_mapping_i1luql2r.csv"
# test側path
SINGLE_RUNS_CSV_B = ARTIFACTS_ROOT / "alpha_sweep_mapping_s9850pgk.csv"

# --- 共通のデータセット名 ---
DATASET = "UNSW-NB15"

# --- 手書きモード用: 実験IDとパス ---
MANUAL_EXPERIMENT = "2026-01-15T01-40-45_incremental_ve5hgbcs"
MANUAL_JSON_PATH = f"experiments/{DATASET}/{MANUAL_EXPERIMENT}/experiment.json"

# 手書きモードでテストに使うブロックID（experiment.json の blocks のキー）
MANUAL_TRAIN_BLOCK = "4"
MANUAL_TEST_BLOCK = "5"

# 手書きモードで使う埋め込みパス（必要に応じて編集）
MANUAL_EMBED_PKL_TRAIN = (
    f"/workspace/experiments/{DATASET}/{MANUAL_EXPERIMENT}/models/model_block_004"
)
MANUAL_EMBED_PKL_TEST = (
    f"/workspace/experiments/{DATASET}/{MANUAL_EXPERIMENT}/models/model_block_005"
)

# seed を回す範囲
SEED_RANGE = range(40, 51)

# train/test split のテストサイズ
TEST_SIZE = 0.2

# --- テストデータ側で異常クラスをランダム・オーバーサンプリングするか ---
USE_TEST_OVERSAMPLING = True   # False にすれば今まで通り一切複製しない
MIN_ATTACK_IN_TEST    = 500     # テスト中の Attack(異常) を最低何件まで増やすか（目安値）

# 特徴量候補（存在しない列は自動スキップ）
USE_COLS_BASE = [
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "sttl",
]

# --- ラベル定義（※ 1=Attack, 0=Benign） ---
ATTACK_LABEL = 1
BENIGN_LABEL = 0


# ============================================================
# 共通ユーティリティ
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
    p = Path(path)
    return torch.load(p, map_location="cpu", weights_only=False)


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
        return mean_vec  # 語彙外は平均ベクトルで埋める
    
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


def attach_srcip_embedding(
    df: pd.DataFrame, wv, mean_vec: np.ndarray, ip_col: str = "srcip"
) -> tuple[pd.DataFrame, List[str]]:
    if ip_col not in df.columns:
        raise KeyError(
            f"{ip_col} 列が見つかりません。現在の列: {list(df.columns)[:20]} ..."
        )
    dim = wv.vectors.shape[1]
    vecs = np.vstack(
        [ip_to_vec(ip, wv, mean_vec) for ip in df[ip_col].astype(str).values]
    )
    emb_cols = [f"src_emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(vecs, columns=emb_cols, index=df.index)
    return pd.concat([df, emb_df], axis=1), emb_cols


def select_existing_columns(df: pd.DataFrame, want: List[str]) -> List[str]:
    exist = [c for c in want if c in df.columns]
    missing = [c for c in want if c not in df.columns]
    if missing:
        print(f"[WARN] 入力CSVに存在しない列をスキップします: {missing}")
    return exist

def random_oversample_minority(
    X: pd.DataFrame,
    y: pd.Series,
    label_pos: int = ATTACK_LABEL,
    min_pos: int = 50,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    評価用に、テストデータ中の陽性クラス（label_pos）を
    ランダム・オーバーサンプリングで複製して増やす。

    ※AUC の理論値は複製しても変わらないが、
      ・極端な不均衡で ROC/PR カーブがギザギザになりすぎる
      ・陽性サンプルが少なすぎて統計的に不安
    といった場合の「安定化用」「エラー回避用」のテクニック。
    """
    y_arr = np.asarray(y)
    pos_idx = np.where(y_arr == label_pos)[0]
    n_pos = len(pos_idx)

    # 陽性が0なら何もできないのでそのまま返す
    if n_pos == 0:
        print("[WARN] テストに Attack が 0 件なので oversampling せずそのまま返します。")
        return X, y

    # 既に十分多ければ何もしない
    if n_pos >= min_pos:
        return X, y

    n_extra = min_pos - n_pos
    rng = np.random.default_rng(random_state)
    extra_idx = rng.choice(pos_idx, size=n_extra, replace=True)

    # 追加分を作る
    X_extra = X.iloc[extra_idx]
    y_extra = y.iloc[extra_idx]

    X_os = pd.concat([X, X_extra], axis=0).reset_index(drop=True)
    y_os = pd.concat([y, y_extra], axis=0).reset_index(drop=True)

    print(
        f"[Oversample] Attack を {n_pos} 件 -> {len(y_os[y_os == label_pos])} 件 に増やしました "
        f"(min_pos={min_pos})"
    )
    return X_os, y_os

def random_undersample_majority(
    X, y,
    label_neg=BENIGN_LABEL,
    max_neg=5000,
    random_state=None,
):
    idx_neg = np.where(y == label_neg)[0]
    if len(idx_neg) <= max_neg:
        return X, y

    rng = np.random.default_rng(random_state)
    drop_idx = rng.choice(idx_neg, size=len(idx_neg) - max_neg, replace=False)

    keep = np.ones(len(y), dtype=bool)
    keep[drop_idx] = False

    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


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

    print(f"[{prefix}] ROC-AUC = {auc:.6f}")
    print(f"[{prefix}] ROC head:")
    for i in range(min(2, len(fpr))):
        print(
            f"  thr={thr[i]:.6f}  fpr={fpr[i]:.6f}  tpr={tpr[i]:.6f}"
        )
    if len(fpr) > 0:
        print(
            f"  ... last: thr={thr[-1]:.6f}  fpr={fpr[-1]:.6f}  tpr={tpr[-1]:.6f}"
        )

    roc_df = pd.DataFrame({"threshold": thr, "fpr": fpr, "tpr": tpr})
    roc_csv = out_dir / f"roc_{prefix}.csv"
    roc_df.to_csv(roc_csv, index=False)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {prefix}")
    plt.legend(loc="lower right")
    roc_png = out_dir / f"roc_{prefix}.png"
    plt.savefig(roc_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{prefix}] ROC CSV: {roc_csv}")
    print(f"[{prefix}] ROC PNG: {roc_png}")


def save_score_count_hist(
    y_true,
    score,
    out_dir: Path,
    prefix: str,
    *,
    bins: int = 50,
    thr: float | None = None,
):
    """
    異常スコア（または Attack 確率）を横軸とし、Attack/Benign の“件数”を同一ビンでカウントして
    積み上げ棒グラフにして保存。あわせてCSV（ビンごとの件数）を出力。
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    if score.size == 0:
        print(f"[{prefix}] empty score array; skip count hist.")
        return
    s_min, s_max = float(np.min(score)), float(np.max(score))
    if s_min == s_max:
        # スコアが全て同じ場合は±1e-6だけ広げる
        s_min -= 1e-6
        s_max += 1e-6
    edges = np.linspace(s_min, s_max, bins + 1)
    cnt_benign, _ = np.histogram(score[y_true == 0], bins=edges)
    cnt_attack, _ = np.histogram(score[y_true == 1], bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0]

    out_csv = Path(out_dir) / f"score_count_hist_{prefix}.csv"
    pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "midpoint": mids,
            "count_benign": cnt_benign,
            "count_attack": cnt_attack,
            "count_total": cnt_benign + cnt_attack,
        }
    ).to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 4))
    plt.bar(
        edges[:-1],
        cnt_benign,
        align="edge",
        width=width,
        alpha=0.75,
        label="Benign (count)",
    )
    plt.bar(
        edges[:-1],
        cnt_attack,
        align="edge",
        width=width,
        alpha=0.75,
        bottom=cnt_benign,
        label="Attack (count)",
    )
    if thr is not None:
        plt.axvline(thr, linestyle="--", linewidth=1.2, label=f"threshold={thr}")
    plt.xlabel("Anomaly score / P(Attack)")
    plt.ylabel("Count")
    plt.title(f"Score count histogram - {prefix}")
    plt.legend()
    out_png = Path(out_dir) / f"score_count_hist_{prefix}.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{prefix}] Score count hist CSV:", out_csv)
    print(f"[{prefix}] Score count hist PNG:", out_png)


# ============================================================
# レポート dataclass
# ============================================================

@dataclass
class RunReport:
    setting: str
    n_train: int
    n_test: int
    roc_auc: float | None
    ap: float | None  # average precision (PR-AUC 近似)
    threshold_desc: str
    confusion: List[List[int]] | None  # [[tn, fp],[fn, tp]] ではなく sklearn の表示順
    notes: str = ""


def save_report_json(report: RunReport, out_dir: Path, fname: str):
    with open(out_dir / fname, "w") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


# ============================================================
# experiment.json 関連（CSVモード用）
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
      - blocks の最後から2番目を train ブロック
      - blocks の最後       を test ブロック
    として、
      - input_csv (test ブロックのCSV)
      - embed_train_path (trainブロックのmodel_path)
      - embed_test_path  (testブロックのmodel_path)
    を返す。
    """
    cfg = load_experiment_config(dataset, run_id)
    blocks = cfg["blocks"]
    results_blocks = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    if len(block_nums) < 3:
        raise ValueError("B4/B5 を使うには最低 3 block 必要")

    train_num = block_nums[-3]   # B4
    test_num  = block_nums[-2]   # B5

    return {
        "train_csv": blocks[str(train_num)],
        "test_csv":  blocks[str(test_num)],
        "embed_train_path": results_blocks[f"{train_num:03d}"]["model"]["model_path"],
        "embed_test_path":  results_blocks[f"{test_num:03d}"]["model"]["model_path"],
    }

def resolve_paths_from_config_single(dataset: str, run_id: str) -> dict:
    """
    single 用（安全版）:
      - input_csv : Attack が含まれる後ろ側 block
      - embed_path: results["blocks"] に存在する最新の model
    """
    cfg = load_experiment_config(dataset, run_id)
    blocks: dict = cfg["blocks"]
    results_blocks: dict = cfg["results"]["blocks"]

    block_nums = sorted(int(k) for k in blocks.keys())
    if len(block_nums) < 2:
        raise ValueError(f"single なのにブロック数が2未満です: {block_nums}")

    # ===== embed_path: 実在する model block の最大 =====
    model_block_nums = sorted(int(k) for k in results_blocks.keys())
    if not model_block_nums:
        raise ValueError("results.blocks に model が1つも存在しません")

    model_num = model_block_nums[-1]
    embed_path = results_blocks[f"{model_num:03d}"]["model"]["model_path"]

    return {
        "csv": blocks[f"{block_nums[-1]}"],
        "embed_path": embed_path,
    }


# ============================================================
# グローバル（各runごとに上書きする）
# ============================================================

df_train: pd.DataFrame
df_test: pd.DataFrame
wv_train = None
wv_test = None
mean_vec_train: np.ndarray
mean_vec_test: np.ndarray
CAT_COLS: List[str] = []
NUM_COLS: List[str] = []
OUT_DIR: Path
CURRENT_RUN_ID: str | None = None
CURRENT_MODE: str | None = None
CURRENT_ALPHA: float | None = None


# ============================================================
# 1 seed 分の IF 実行
# ============================================================

def run_isoforest_for_seed(seed: int) -> dict:
    """
    1つの seed について IF を学習・評価し、
    AUC などのメトリクスと保存先を dict で返す。
    出力は OUT_DIR/seed_{seed}/ 以下にまとめる。
    """
    global df_train, df_test, wv_train, wv_test, mean_vec_train, mean_vec_test
    global CAT_COLS, NUM_COLS, OUT_DIR
    global CURRENT_RUN_ID, CURRENT_MODE, CURRENT_ALPHA

    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir_seed = ensure_outdir(Path(OUT_DIR) / f"seed_{seed}")

    # --- 特徴量とラベルを作成 ---
    need_cols = ["Label", "srcip"] + CAT_COLS + NUM_COLS

    # ==========
    # B4 → train
    # ==========
    work_train = df_train[need_cols].copy()
    X_train = work_train.drop(columns=["Label"])
    y_train = work_train["Label"].astype(int)

    # ==========
    # B5 → test
    # ==========
    work_test = df_test[need_cols].copy()
    X_test = work_test.drop(columns=["Label"])
    y_test = work_test["Label"].astype(int)

    X_test, y_test = random_undersample_majority(
        X_test, y_test,
        label_neg=BENIGN_LABEL,
        max_neg=10000,
        random_state=seed,
    )


    # ★ テストデータ側の Attack をランダム・オーバーサンプリング（任意）
    if USE_TEST_OVERSAMPLING:
        X_test, y_test = random_oversample_minority(
            X_test,
            y_test,
            label_pos=ATTACK_LABEL,
            min_pos=MIN_ATTACK_IN_TEST,
            random_state=seed,
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

    #breakpoint()

    iso_clf = IsolationForest(
        n_estimators=1000,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )

    pipe_iso = Pipeline(
        [
            ("prep", preprocess),
            ("clf", iso_clf),
        ]
    )

    pipe_iso.fit(X_train_benign)

    # --- スコア & メトリクス ---
    dec = pipe_iso.decision_function(X_test)  # 高いほど正常
    anom_score = -dec  # 高いほど異常

    # ===== 検証② permutation =====
    base_auc, perm = permutation_importance_group(
        pipe_iso,
        X_test,
        y_test,
        {"embedding": emb_cols, "non_embedding": CAT_COLS + NUM_COLS},
    )
    with open(out_dir_seed / "permutation_importance_group.json", "w") as f:
        json.dump({"base_auc": base_auc, "groups": perm}, f, indent=2)

    from sklearn.tree import plot_tree
    #一つの木を可視化
    estimator = iso_clf.estimators_[0]
    plt.figure(figsize=(20,10))

    feature_names = pipe_iso.named_steps['prep'].get_feature_names_out()
    plot_tree(estimator, filled=True, rounded=True, feature_names=feature_names)
    plt.savefig(out_dir_seed / "isoforest_tree_0.png")
    plt.close()

    try:
        roc_if = roc_auc_score(y_test, anom_score)
    except ValueError:
        roc_if = None

    try:
        ap_if = average_precision_score(y_test, anom_score)
    except ValueError:
        ap_if = None

    thr = 0.0
    y_pred_if = (anom_score > thr).astype(int)
    cm_if = confusion_matrix(y_test, y_pred_if).tolist()

    setting = f"IF_seed_{seed}"
    if CURRENT_RUN_ID is not None:
        setting = f"IF_run={CURRENT_RUN_ID}_mode={CURRENT_MODE}_alpha={CURRENT_ALPHA}_seed={seed}"

    notes = f"seed={seed}, CAT={CAT_COLS}, NUM={NUM_COLS}, EMBED_DIM={len(emb_cols)}"
    if CURRENT_RUN_ID is not None:
        notes = (
            f"run_id={CURRENT_RUN_ID}, mode={CURRENT_MODE}, alpha_anom={CURRENT_ALPHA}, "
            + notes
        )

    rep_if = RunReport(
        setting=setting,
        n_train=int(mask_benign_train.sum()),
        n_test=len(y_test),
        roc_auc=roc_if,
        ap=ap_if,
        threshold_desc="decision_function<0 を異常（異常スコア>0）として判定",
        confusion=cm_if,
        notes=notes,
    )
    save_report_json(rep_if, out_dir_seed, "report_isoforest.json")

    pred_if = pd.DataFrame(
        {
            "y_true": y_test.values,
            "anom_score": anom_score,
            "y_pred": y_pred_if,
        }
    )
    pred_if.to_csv(out_dir_seed / "pred_isoforest.csv", index=False)

    print(f"[seed={seed}] ROC-AUC:", roc_if, " AP:", ap_if)
    print(f"[seed={seed}] Confusion matrix:\n", np.array(cm_if))

    # ROC曲線＆ヒストグラム
    prefix = f"isoforest_seed{seed}"
    if CURRENT_RUN_ID is not None:
        prefix = f"isoforest_{CURRENT_RUN_ID}_seed{seed}"

    save_and_print_roc(y_test.values, anom_score, out_dir_seed, prefix=prefix)
    save_score_count_hist(
        y_test.values,
        anom_score,
        out_dir_seed,
        prefix=prefix,
        bins=60,
        thr=thr,
    )

    return {
        "seed": seed,
        "n_train": int(mask_benign_train.sum()),
        "n_test": len(y_test),
        "roc_auc": roc_if,
        "ap": ap_if,
    }


def run_seeds_and_save_summary(seeds: List[int]) -> None:
    global OUT_DIR

    results = []
    for s in seeds:
        print(f"\n===== run seed={s} =====")
        res = run_isoforest_for_seed(s)
        results.append(res)

    results_df = pd.DataFrame(results)
    results_csv = Path(OUT_DIR) / "isoforest_seed_results.csv"
    results_df.to_csv(results_csv, index=False)
    print("Per-seed results CSV:", results_csv)

    aucs = [r["roc_auc"] for r in results if r["roc_auc"] is not None]
    if aucs:
        auc_mean = float(np.mean(aucs))
        auc_max = float(np.max(aucs))
        auc_min = float(np.min(aucs))
        auc_med = float(np.median(aucs))
        auc_std = float(np.std(aucs))
    else:
        auc_mean = auc_max = auc_min = auc_med = auc_std = None

    summary = {
        "seeds": list(seeds),
        "n_runs": len(aucs),
        "roc_auc_mean": auc_mean,
        "roc_auc_max": auc_max,
        "roc_auc_min": auc_min,
        "roc_auc_median": auc_med,
        "roc_auc_std": auc_std,
    }

    summary_json = Path(OUT_DIR) / "isoforest_seed_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("AUC summary JSON:", summary_json)
    print(
        "AUC stats:",
        "mean=",
        auc_mean,
        "max=",
        auc_max,
        "min=",
        auc_min,
        "median=",
        auc_med,
        "std=",
        auc_std,
    )


# ============================================================
# 手書きモード
# ============================================================

def main_manual():
    """
    これまで通り、MANUAL_EXPERIMENT で1本だけ評価するモード
    """
    global df_train, df_test, wv_train, wv_test, mean_vec_train, mean_vec_test
    global CAT_COLS, NUM_COLS, OUT_DIR
    global CURRENT_RUN_ID, CURRENT_MODE, CURRENT_ALPHA

    CURRENT_RUN_ID = MANUAL_EXPERIMENT
    CURRENT_MODE = "manual"
    CURRENT_ALPHA = None

    with open(MANUAL_JSON_PATH, "r") as f:
        config = json.load(f)

    train_csv = config["blocks"][MANUAL_TRAIN_BLOCK]
    test_csv  = config["blocks"][MANUAL_TEST_BLOCK]
    embed_train_path = MANUAL_EMBED_PKL_TRAIN
    embed_test_path = MANUAL_EMBED_PKL_TEST
    
    df_train = load_csv(train_csv)
    df_test = load_csv(test_csv)

    rand8 = "".join(
        np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=8)
    )
    OUT_DIR = ensure_outdir(f"eval/eval_anomaly_{MANUAL_EXPERIMENT}_{rand8}")
    print("OUT_DIR:", OUT_DIR)

    model_train = load_embeddings(embed_train_path)
    model_test = load_embeddings(embed_test_path)
    wv_train = get_embedding_interface(model_train)
    wv_test = get_embedding_interface(model_test)
    mean_vec_train = compute_mean_vector(wv_train)
    mean_vec_test = compute_mean_vector(wv_test)

    use_cols = select_existing_columns(df_train, USE_COLS_BASE)
    CAT_COLS = [c for c in ["proto", "state", "service"] if c in use_cols]
    NUM_COLS = [c for c in use_cols if c not in CAT_COLS]

    print("USE_COLS:", use_cols)
    print("CAT_COLS:", CAT_COLS)
    print("NUM_COLS:", NUM_COLS)

    run_seeds_and_save_summary(list(SEED_RANGE))


# ============================================================
# CSVモード（mode == incremental の行だけ回す）
# ============================================================

def main_from_csv_incremental():
    """
    alpha_anom,mode,run_id のCSVを読み込み、
    mode == "incremental" の行のみ run_id ごとに評価するモード
    """
    global df_train, df_test, wv_train, wv_test, mean_vec_train, mean_vec_test
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

        df_train = load_csv(paths["train_csv"])
        df_test  = load_csv(paths["test_csv"])

        embed_train_path = paths["embed_train_path"]
        embed_test_path = paths["embed_test_path"]

        # run_id 単位で出力ディレクトリを分ける
        OUT_DIR = ensure_outdir(Path("eval") / OUT_DIR_NAME / run_id)
        print("OUT_DIR:", OUT_DIR)

        model_train = load_embeddings(embed_train_path)
        model_test = load_embeddings(embed_test_path)
        wv_train = get_embedding_interface(model_train)
        wv_test = get_embedding_interface(model_test)
        mean_vec_train = compute_mean_vector(wv_train)
        mean_vec_test = compute_mean_vector(wv_test)

        use_cols = select_existing_columns(df_train, USE_COLS_BASE)
        CAT_COLS = [c for c in ["proto", "state", "service"] if c in use_cols]
        NUM_COLS = [c for c in use_cols if c not in CAT_COLS]

        print("USE_COLS:", use_cols)
        print("CAT_COLS:", CAT_COLS)
        print("NUM_COLS:", NUM_COLS)

        run_seeds_and_save_summary(list(SEED_RANGE))

def main_from_csv_single():
    """
    single モード：
    - 2つの CSV（A, B）を読み込む
    - 各 CSV の run_id を上から順にペアリング
    - それぞれ resolve_paths_from_config_incremental() に通して
        embed_train_path / embed_test_path を得る
    - A 側の input_csv を評価対象として使用
    """
    global df_train, df_test, wv_train, wv_test, mean_vec_train, mean_vec_test
    global CAT_COLS, NUM_COLS, OUT_DIR
    global CURRENT_RUN_ID, CURRENT_MODE, CURRENT_ALPHA

    # =========================
    # CSV 読み込み
    # =========================
    df_a = pd.read_csv(SINGLE_RUNS_CSV_A)
    df_b = pd.read_csv(SINGLE_RUNS_CSV_B)

    if "run_id" not in df_a.columns:
        raise ValueError("CSV_A に run_id 列がありません")
    if "run_id" not in df_b.columns:
        raise ValueError("CSV_B に run_id 列がありません")

    if len(df_a) != len(df_b):
        raise ValueError(
            f"CSV_A({len(df_a)}) と CSV_B({len(df_b)}) の行数が一致しません"
        )

    print(f"[INFO] single mode: {len(df_a)} pairs")

    # =========================
    # 上から順に処理
    # =========================
    for idx in range(len(df_a)):
        run_id_a = str(df_a.iloc[idx]["run_id"])
        run_id_b = str(df_b.iloc[idx]["run_id"])

        CURRENT_RUN_ID = f"single_{idx:03d}"
        CURRENT_MODE = "single"
        CURRENT_ALPHA = None

        print(
            f"\n########## SINGLE idx={idx} ##########\n"
            f"run_id_A = {run_id_a}\n"
            f"run_id_B = {run_id_b}"
        )

        # =========================
        # run_id → パス変換
        # =========================
        paths_a = resolve_paths_from_config_single(DATASET, run_id_a)
        paths_b = resolve_paths_from_config_single(DATASET, run_id_b)

        train_csv_a = paths_a["csv"]
        train_csv_b = paths_b["csv"]
        df_train = load_csv(train_csv_a)
        df_test = load_csv(train_csv_b)

        # A 側を train embedding に使用
        embed_train_path = paths_a["embed_path"]
        embed_test_path = paths_b["embed_path"]

        print("input_csv (from train):", train_csv_a)
        print("input_csv (from test):", train_csv_b)
        print("embed_train_path:", embed_train_path)
        print("embed_test_path:", embed_test_path)

        # =========================
        # 出力ディレクトリ
        # =========================
        OUT_DIR = ensure_outdir(
            Path("eval")
            / OUT_DIR_NAME
            / f"single_{idx:03d}_A_{run_id_a}_B_{run_id_b}"
        )
        print("OUT_DIR:", OUT_DIR)

        # =========================
        # 埋め込みロード
        # =========================
        model_train = load_embeddings(embed_train_path)
        model_test = load_embeddings(embed_test_path)

        wv_train = get_embedding_interface(model_train)
        wv_test = get_embedding_interface(model_test)

        mean_vec_train = compute_mean_vector(wv_train)
        mean_vec_test = compute_mean_vector(wv_test)

        # =========================
        # 特徴量選択
        # =========================
        use_cols = select_existing_columns(df_train, USE_COLS_BASE)
        CAT_COLS = [c for c in ["proto", "state", "service"] if c in use_cols]
        NUM_COLS = [c for c in use_cols if c not in CAT_COLS]

        print("USE_COLS:", use_cols)
        print("CAT_COLS:", CAT_COLS)
        print("NUM_COLS:", NUM_COLS)

        # =========================
        # IF 実行
        # =========================
        run_seeds_and_save_summary(list(SEED_RANGE))



# ============================================================
# エントリポイント
# ============================================================

if __name__ == "__main__":
    if USE_RUNS_CSV:
        if mode_single:
            main_from_csv_single()
        else:
            main_from_csv_incremental()
    else:
        main_manual()
