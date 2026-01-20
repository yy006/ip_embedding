import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ===============================
# 設定
# ===============================

ARTIFACTS_ROOT = Path("/workspace")
CSV_PATH = ARTIFACTS_ROOT / "results" / "attack_block_mode_mean_metrics_UNSW-NB15.csv"
OUT_DIR = ARTIFACTS_ROOT / "results" / "plots_by_attack"
OUT_DIR.mkdir(exist_ok=True)

METRIC = "mean_knn1_auc"   # "mean_ap" に変えてもOK

# ===============================
# 読み込み
# ===============================
df = pd.read_csv(CSV_PATH)

required_cols = {"attack", "block_num", "mode", METRIC}
assert required_cols.issubset(df.columns), df.columns

# ===============================
# attack ごとに描画
# ===============================
for attack, df_a in df.groupby("attack"):
    plt.figure(figsize=(6, 4))

    for mode in ["single", "incremental"]:
        df_m = df_a[df_a["mode"] == mode].sort_values("block_num")
        if df_m.empty:
            continue

        plt.plot(
            df_m["block_num"],
            df_m[METRIC],
            marker="o",
            label=mode,
        )
    plt.ylim(0.3, 1.0)
    plt.yticks(np.arange(0.3, 1.01, 0.1))

    plt.title(f"{attack}: {METRIC} vs block")
    plt.xlabel("Block number")
    plt.ylabel(METRIC)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"{attack}_{METRIC}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] saved: {out_path}")
