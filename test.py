# quick_profile.py
import pandas as pd, numpy as np, sys, pathlib
from datetime import timedelta

path = sys.argv[1]  # 入力ファイル（.csv/.parquet/.tsv など）
ext = pathlib.Path(path).suffix.lower()

# 1) 読み込み（拡張子で自動）
if ext in [".parquet"]:
    df = pd.read_parquet(path)
elif ext in [".tsv"]:
    df = pd.read_csv(path, sep="\t", low_memory=False)
else:
    df = pd.read_csv(path, low_memory=False)

# 2) 基本プロファイル
print("=== SHAPE ===", df.shape)
print("=== DTYPE COUNTS ===", df.dtypes.value_counts())
print("\n=== HEAD ===")
print(df.head(3))

# 3) タイムスタンプ推定
ts_candidates = [c for c in df.columns if "time" in c.lower() or "ts" == c.lower() or "timestamp" in c.lower()]
if ts_candidates:
    ts = ts_candidates[0]
    df[ts] = pd.to_datetime(df[ts], errors="coerce", utc=True)
    print(f"\n[time] column: {ts}  min={df[ts].min()}  max={df[ts].max()}  null%={(df[ts].isna().mean()*100):.2f}")
    # 並び・ギャップ
    s = df[ts].dropna().sort_values()
    if len(s) > 1:
        gaps = s.diff().dropna()
        print(f"median_gap={gaps.median()},  95p_gap={gaps.quantile(0.95)}")

# 4) 欠損・一意性・カードinality（上位10列）
cols = df.columns[:10]
miss = df[cols].isna().mean().sort_values(ascending=False).head(10)
card = df[cols].nunique().sort_values(ascending=False).head(10)
print("\n=== MISSING RATE (top10) ===\n", miss)
print("\n=== CARDINALITY (top10) ===\n", card)

# 5) ネットワークっぽい列の分布
guess_ip = [c for c in df.columns if "ip" in c.lower()]
guess_port = [c for c in df.columns if "port" in c.lower()]
guess_proto = [c for c in df.columns if "proto" in c.lower()]

def top_counts(col, n=10):
    if col in df.columns:
        print(f"\nTOP {n} of {col}")
        print(df[col].value_counts(dropna=False).head(n))

for c in guess_ip + guess_port + guess_proto:
    top_counts(c)

# 時間×プロトコルの粗い頻度（5分バケット）
if ts_candidates:
    ts = ts_candidates[0]
    if "proto" in df.columns:
        q = f"""
        SELECT date_trunc('minute', {ts}) as m, proto, COUNT(*) as n
        FROM t
        WHERE {ts} IS NOT NULL
        GROUP BY 1,2
        ORDER BY 1
        """
        out = con.execute(q).fetch_df()
        print("\n=== per-minute counts by proto (head) ===")
        print(out.head(10))
