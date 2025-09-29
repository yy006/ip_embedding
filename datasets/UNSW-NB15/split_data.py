from pathlib import Path
import pandas as pd

in_csv   = "5tuple_data/UNSW-NB15_2_5tuple.csv"             # 入力
out_dir  = Path("by2h")                     # 出力先ディレクトリ
prefix   = "datasetX"                       # ファイル名プレフィックス（任意に変更）

out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_csv)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"])

# 2時間に丸めたキーを作成（00,02,04,...）
df["bin2h"] = df["Timestamp"].dt.floor("2H")

# 各2時間ビンごとに出力
for bin_ts, g in df.groupby("bin2h", sort=True):
    start = pd.to_datetime(bin_ts)
    end   = start + pd.Timedelta(hours=2)
    fname = f"{prefix}_{start.strftime('%Y%m%d%H')}_{end.strftime('%Y%m%d%H')}_by2h.csv"
    g.drop(columns=["bin2h"]).to_csv(out_dir / fname, index=False)
    print(f"wrote: {out_dir/fname}  rows={len(g)}")
