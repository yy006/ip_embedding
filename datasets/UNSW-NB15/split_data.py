from pathlib import Path
import pandas as pd

# 何時間ごとに分割するか
k = 1
HERE = Path(__file__).resolve().parent
in_csv   = HERE/"UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h"/"2015021802_2015021804_by2h.csv"             # 入力
out_dir  = Path(f"by{k}h")                     # 出力先ディレクトリ
prefix   = "datasetX"                       # ファイル名プレフィックス（任意に変更）

out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_csv)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"])

# k時間に丸めたキーを作成（00,02,04,...）
df[f"bin{k}h"] = df["Timestamp"].dt.floor(f"{k}H")

# 各k時間ビンごとに出力
for bin_ts, g in df.groupby(f"bin{k}h", sort=True):
    start = pd.to_datetime(bin_ts)
    end   = start + pd.Timedelta(hours=k)
    fname = f"{prefix}_{start.strftime('%Y%m%d%H')}_{end.strftime('%Y%m%d%H')}_by{k}h.csv"
    g.drop(columns=[f"bin{k}h"]).to_csv(out_dir / fname, index=False)
    print(f"wrote: {out_dir/fname}  rows={len(g)}")
