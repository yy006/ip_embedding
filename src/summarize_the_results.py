from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 設定
# =========================
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = Path(ROOT) / "eval"
#BASE_DIR = ARTIFACTS_ROOT / "ノルム制約各攻撃_single_攻撃多い_12次元_dupfalse"
#CONFIG_CSV = BASE_DIR / "alpha_sweep_mapping_s9850pgk.csv"
BASE_DIR = ARTIFACTS_ROOT / "ノルム制約各攻撃_incremental_攻撃多い_12次元_dupfalse"
CONFIG_CSV = BASE_DIR / "alpha_sweep_mapping_lg2rtcy0.csv"
OUT_DIR = BASE_DIR / "summary"
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = OUT_DIR / "experiment_summary.csv"
FIG_PATH = OUT_DIR / "auc_vs_alpha.png"
FIG_PATH_RADIUS = OUT_DIR / "auc_vs_radius.png"

# =========================
# 1. 実験設定を読む
# =========================
config_df = pd.read_csv(CONFIG_CSV)

# run_id を文字列として扱う
config_df["run_id"] = config_df["run_id"].astype(str)

# =========================
# 2. 各 run ディレクトリから結果を回収
# =========================
records = []

for run_dir in BASE_DIR.iterdir():
    if not run_dir.is_dir():
        continue

    summary_json = run_dir / "isoforest_seed_summary.json"
    if not summary_json.exists():
        continue

    # run_id をディレクトリ名から特定
    matched = None
    for rid in config_df["run_id"]:
        if rid in run_dir.name:
            matched = rid
            break

    if matched is None:
        print(f"[WARN] run_id が見つからない: {run_dir.name}")
        continue

    # 結果 json 読み込み
    with open(summary_json, "r", encoding="utf-8") as f:
        result = json.load(f)

    # 対応する実験設定を取得
    cfg = config_df[config_df["run_id"] == matched].iloc[0]

    record = {
        # ---- 実験設定 ----
        "run_id": matched,
        "alpha_anom": cfg.get("alpha_anom"),
        "mode": cfg.get("mode"),
        "attack": cfg.get("attack"),
        "Radius": cfg.get("Radius"),
        "normal_pull_lambda": cfg.get("normal_pull_lambda"),

        # ---- 実験結果 ----
        "roc_auc_mean": result.get("roc_auc_mean"),
        "roc_auc_max": result.get("roc_auc_max"),
        "roc_auc_min": result.get("roc_auc_min"),
        "roc_auc_median": result.get("roc_auc_median"),
        "roc_auc_std": result.get("roc_auc_std")
    }

    records.append(record)

# =========================
# 3. DataFrame 化 & 保存
# =========================
summary_df = pd.DataFrame(records)
summary_df = summary_df.sort_values(
    ["attack", "mode", "alpha_anom", "Radius", "normal_pull_lambda"],
    ignore_index=True
)

summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"[OK] summary saved -> {SUMMARY_CSV}")

# =========================
# 4. グラフ作成（例：AUC vs alpha）
# =========================
plt.figure(figsize=(6, 4))

for (attack, mode), g in summary_df.groupby(["attack", "mode"]):
    plt.plot(
        g["alpha_anom"],
        g["roc_auc_mean"],
        marker="o",
        label=f"{attack}-{mode}",
    )

plt.xlabel("alpha_anom")
plt.ylabel("AUC")
plt.ylim(0.4, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close()

print(f"[OK] figure saved -> {FIG_PATH}")

# =========================
# 5. グラフ作成（AUC vs Radius, attackごと）
# =========================
df_r = summary_df.copy()

# attack を分けてプロットするかどうかの判定
use_attack = (
    "attack" in df_r.columns and
    df_r["attack"].notna().any() and
    df_r["attack"].nunique() > 1
)

df_r["Radius"] = df_r["Radius"].fillna(0)

if use_attack:
    for attack, df_attack in df_r.groupby("attack"):
        plt.figure(figsize=(6, 4))

        for (mode, alpha), g in df_attack.groupby(["mode", "alpha_anom"]):
            g = g.sort_values("Radius")
            plt.scatter(
                g["Radius"],
                g["roc_auc_mean"],
                label=f"{mode}-α={alpha}",
                s=60,
                alpha=0.8,
            )

        plt.xlabel("Radius")
        plt.ylabel("AUC")
        plt.ylim(0.4, 1.0)
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.title(f"AUC vs Radius ({attack})")
        plt.tight_layout()

        fig_path = OUT_DIR / f"auc_vs_radius_attack={attack}.png"
        plt.savefig(fig_path)
        plt.close()

        print(f"[OK] figure saved -> {fig_path}")

else:
    plt.figure(figsize=(6, 4))

    for (mode, alpha), g in df_r.groupby(["mode", "alpha_anom"]):
        g = g.sort_values("Radius")
        plt.scatter(
            g["Radius"],
            g["roc_auc_mean"],
            label=f"{mode}-α={alpha}",
            s=60,
            alpha=0.8,
        )

    plt.xlabel("Radius")
    plt.ylabel("AUC")
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.title("AUC vs Radius (all attacks)")
    plt.tight_layout()

    fig_path = OUT_DIR / "auc_vs_radius_all_attacks.png"
    plt.savefig(fig_path)
    plt.close()

    print(f"[OK] figure saved -> {fig_path}")


# =========================
# Radius ごとに平均を取った DataFrame
# =========================
df_r = summary_df.copy()

# attack を分けてプロットするかどうかの判定
use_attack = (
    "attack" in df_r.columns and
    df_r["attack"].notna().any() and
    df_r["attack"].nunique() > 1
)

df_r["Radius"] = df_r["Radius"].fillna(0)

# =========================
# 6. グラフ作成（AUC vs Radius, 平均1点, attackごと）
# =========================
if use_attack:
    df_radius_mean = (
        df_r
        .groupby(["attack", "mode", "alpha_anom", "Radius"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc_mean", "mean"),
            roc_auc_std=("roc_auc_std", "mean"),   # std の平均（簡易）
            n_points=("roc_auc_mean", "count")
        )
    )

    for attack, df_attack in df_radius_mean.groupby("attack"):
        plt.figure(figsize=(6, 4))

        for (mode, alpha), g in df_attack.groupby(["mode", "alpha_anom"]):
            plt.scatter(
                g["Radius"],
                g["roc_auc_mean"],
                label=f"{mode}-α={alpha}",
                s=80,
            )

        plt.xlabel("Radius")
        plt.ylabel("AUC")
        plt.ylim(0.4, 1.0)
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.title(f"AUC vs Radius (mean, {attack})")
        plt.tight_layout()

        fig_path = OUT_DIR / f"auc_vs_radius_mean_attack={attack}.png"
        plt.savefig(fig_path)
        plt.close()

        print(f"[OK] figure saved -> {fig_path}")

else:
    df_radius_mean = (
        df_r
        .groupby(["mode", "alpha_anom", "Radius"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc_mean", "mean"),
            roc_auc_std=("roc_auc_std", "mean"),   # std の平均（簡易）
            n_points=("roc_auc_mean", "count")
        )
    )

    plt.figure(figsize=(6, 4))

    for (mode, alpha), g in df_radius_mean.groupby(["mode", "alpha_anom"]):
        plt.scatter(
            g["Radius"],
            g["roc_auc_mean"],
            label=f"{mode}-α={alpha}",
            s=80,
        )

    plt.xlabel("Radius")
    plt.ylabel("AUC")
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.title("AUC vs Radius (mean, all attacks)")
    plt.tight_layout()

    fig_path = OUT_DIR / "auc_vs_radius_mean_all_attacks.png"
    plt.savefig(fig_path)
    plt.close()

    print(f"[OK] figure saved -> {fig_path}")

# =========================
# 7. グラフ作成（AUC vs normal_pull_lambda, 平均, R固定, attackごと）
# =========================
"""
FIXED_RADIUS_LIST = [None, 0.25, 0.5, 1.0, 2.0]

LAMBDA_ORDER = [0, 0.001, 0.005, 0.01, 0.1]
lambda_to_x = {v: i for i, v in enumerate(LAMBDA_ORDER)}

# ---- ① 平均を取る ----
df_lp_mean = (
    summary_df
    .groupby(
        ["attack", "mode", "normal_pull_lambda", "Radius"],
        as_index=False
    )
    .agg(
        roc_auc_mean=("roc_auc_mean", "mean"),
        roc_auc_std=("roc_auc_mean", "std"),
        n_runs=("roc_auc_mean", "count"),
    )
)

for FIXED_RADIUS in FIXED_RADIUS_LIST:

    df_plot = df_lp_mean.copy()
    df_plot["Radius"] = df_plot["Radius"].astype(float)
    df_plot["normal_pull_lambda"] = df_plot["normal_pull_lambda"].astype(float)

    # R を固定（None の場合は全体）
    if FIXED_RADIUS is not None:
        df_plot = df_plot[df_plot["Radius"] == FIXED_RADIUS]

    if df_plot.empty:
        print(f"[WARN] Radius={FIXED_RADIUS} のデータが存在しません")
        continue

    for attack, df_attack in df_plot.groupby("attack"):
        plt.figure(figsize=(6, 4))

        for mode, g in df_attack.groupby("mode"):
            xs, ys = [], []

            for v, auc in zip(
                g["normal_pull_lambda"],
                g["roc_auc_mean"]
            ):
                v_round = round(v, 6)
                if v_round in lambda_to_x:
                    xs.append(lambda_to_x[v_round])
                    ys.append(auc)

            plt.scatter(
                xs,
                ys,
                label=mode,
                s=90,
            )

        # ---- 軸設定 ----
        plt.xticks(
            range(len(LAMBDA_ORDER)),
            [str(v) for v in LAMBDA_ORDER]
        )
        plt.xlabel("normal_pull_lambda")
        plt.ylabel("AUC")
        plt.ylim(0.4, 1.0)
        plt.grid(True, axis="y")
        plt.legend()
        plt.title(f"AUC vs normal_pull_lambda (mean, R={FIXED_RADIUS}, {attack})")
        plt.tight_layout()

        fig_path = (
            OUT_DIR
            / f"auc_vs_pull_lambda_mean_R={FIXED_RADIUS}_attack={attack}.png"
        )
        plt.savefig(fig_path)
        plt.close()

        print(f"[OK] figure saved -> {fig_path}")

"""