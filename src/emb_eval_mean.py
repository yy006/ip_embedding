import pandas as pd
from pathlib import Path
from typing import List, Dict

# ===============================
# 設定
# ===============================
ARTIFACTS_ROOT = Path("/workspace")

OUT_CSV = ARTIFACTS_ROOT / "results" / "attack_block_mode_mean_metrics_UNSW-NB15.csv"

# eval CSV に含まれる精度列
METRIC_COLS = ["knn1_auc","knn1_ap","knn1_recall_at_num_attack","knn2_auc","knn2_ap","knn2_recall_at_num_attack","knn3_auc","knn3_ap","knn3_recall_at_num_attack"]

# ===============================
# 入力定義（ここだけ編集すればOK）
# ===============================
EXPERIMENT_INPUTS: List[Dict] = [
    {
        "alpha_sweep_csv": ARTIFACTS_ROOT / "experiments" / "alpha_sweep_mapping_udoinuzj.csv",
        "eval_csv": ARTIFACTS_ROOT / "eval" / "LOOCV_inc B5" / "ip_embedding_eval_summary.csv",
        "block_num": 5,
        "mode": "incremental",
    },
    {
        "alpha_sweep_csv": ARTIFACTS_ROOT / "experiments" / "alpha_sweep_mapping_mptnpe6e.csv",
        "eval_csv": ARTIFACTS_ROOT / "eval" / "LOOCV_50dupfalse_inc_B13" / "ip_embedding_eval_summary.csv",
        "block_num": 13,
        "mode": "incremental",
    },
    {
        "alpha_sweep_csv": ARTIFACTS_ROOT / "experiments" / "alpha_sweep_mapping_sbsmbu1v.csv",
        "eval_csv": ARTIFACTS_ROOT / "eval" / "LOOCV_50dupfalse_inc_B17" / "ip_embedding_eval_summary.csv",
        "block_num": 17,
        "mode": "incremental",
    },
    {
        "alpha_sweep_csv": ARTIFACTS_ROOT / "experiments" / "alpha_sweep_mapping_omxoncdd.csv",
        "eval_csv": ARTIFACTS_ROOT / "eval" / "LOOCV_50dupfalse_sin_B5" / "ip_embedding_eval_summary.csv",
        "block_num": 5,
        "mode": "single",
    },
    {
        "alpha_sweep_csv": ARTIFACTS_ROOT / "experiments" / "alpha_sweep_mapping_pje4pcpq.csv",
        "eval_csv": ARTIFACTS_ROOT / "eval" / "LOOCV_50dupfalse_sin_B9" / "ip_embedding_eval_summary.csv",
        "block_num": 9,
        "mode": "single",
    },
    {
        "alpha_sweep_csv": ARTIFACTS_ROOT / "experiments" / "alpha_sweep_mapping_f10tpq3l.csv",
        "eval_csv": ARTIFACTS_ROOT / "eval" / "LOOCV_50dupfalse_sin_B13" / "ip_embedding_eval_summary.csv",
        "block_num": 13,
        "mode": "single",
    },
    {
        "alpha_sweep_csv": ARTIFACTS_ROOT / "experiments" / "alpha_sweep_mapping_h2s6qg11.csv",
        "eval_csv": ARTIFACTS_ROOT / "eval" / "LOOCV_50dupfalse_sin_B17" / "ip_embedding_eval_summary.csv",
        "block_num": 17,
        "mode": "single",
    }
]

# ===============================
# メイン処理
# ===============================
dfs = []

for exp in EXPERIMENT_INPUTS:
    df_alpha = pd.read_csv(exp["alpha_sweep_csv"])
    df_eval = pd.read_csv(exp["eval_csv"])

    # 必須列チェック
    if "run_id" not in df_alpha.columns:
        raise ValueError(f"run_id not found in {exp['alpha_sweep_csv']}")
    if "run_id" not in df_eval.columns:
        raise ValueError(f"run_id not found in {exp['eval_csv']}")

    # マージ
    df = (
        df_eval
        .merge(
            df_alpha[["run_id", "attack"]],
            on="run_id",
            how="left",
        )
    )

    # 実験条件を付与
    df["block_num"] = exp["block_num"]
    df["mode"] = exp["mode"]

    # 欠損除外
    df = df.dropna(subset=["attack"])

    dfs.append(df)

# ===============================
# 全実験を結合
# ===============================
df_all = pd.concat(dfs, ignore_index=True)

# ===============================
# 平均集計
# ===============================
group_cols = ["attack", "block_num", "mode"]
agg_dict = {m: "mean" for m in METRIC_COLS}

df_mean = (
    df_all
    .groupby(group_cols, as_index=False)
    .agg(agg_dict)
    .rename(columns={m: f"mean_{m}" for m in METRIC_COLS})
)

# ===============================
# 保存
# ===============================
df_mean.to_csv(OUT_CSV, index=False)

print(f"[OK] saved to {OUT_CSV}")
print(df_mean.head())
