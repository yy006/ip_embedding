import pandas as pd

# 入力
in_csv  = 'UNSW-NB15_2_5tuple_by2h/2015021802_2015021804_by2h.csv'
#in_csv  = "UNSW-NB15_2_5tuple_by2h_ipmap59to175_drop175benign/2015012218_2015012220_by2h.csv"
out_csv = "out.csv"

ip_col    = "Src IP Addr"
label_col = "class"

import pandas as pd
from pathlib import Path

# 対象IPセット
DEL_IPS = {f"175.45.176.{i}" for i in range(4)}         # 175.45.176.0~3
SRC_IPS = {f"59.166.0.{i}"   for i in range(4)}         # 59.166.0.0~3
MAP_59_TO_175 = {f"59.166.0.{i}": f"175.45.176.{i}" for i in range(4)}

# ========= 読み込み =========
df = pd.read_csv(in_csv, low_memory=False)
df.columns = df.columns.str.strip()
for c in (ip_col, label_col):
    if c not in df.columns:
        raise KeyError(f"列が見つかりません: {c}")

df[ip_col] = df[ip_col].astype("string")
df[label_col] = df[label_col].astype("string")

# ========= BEFORE（状況）=========
n_before = len(df)
n_del_targets_before = ((df[ip_col].isin(DEL_IPS)) & (df[label_col].str.casefold()=="benign")).sum()
n_src_targets_before = df[ip_col].isin(SRC_IPS).sum()

print("=== BEFORE ===")
print(f"rows total: {n_before}")
print(f"delete-target (175.45.176.0~3 & class==Benign): {n_del_targets_before}")
print(f"replace-target (59.166.0.0~3): {n_src_targets_before}\n")

# ========= 2) 削除（175.45.176.0~3 & class==Benign）=========
mask_delete = (df[ip_col].isin(DEL_IPS)) & (df[label_col].str.casefold() == "benign")
n_to_drop = mask_delete.sum()
df = df.loc[~mask_delete].copy()

# ========= 1) 置換（59.166.0.i -> 175.45.176.i）=========
df[ip_col] = df[ip_col].replace(MAP_59_TO_175)

# 置換後の確認
n_src_left = df[ip_col].isin(SRC_IPS).sum()
print("After replace: residual 59.166.0.0~3 rows (should be 0):", n_src_left)

# ========= AFTER（最終検証）=========
n_after = len(df)
leftover_del = ((df[ip_col].isin(DEL_IPS)) & (df[label_col].str.casefold()=="benign")).sum()

post_counts_175 = df[df[ip_col].isin(DEL_IPS)][ip_col].value_counts().sort_index()

print("\n=== AFTER ===")
print(f"rows total: {n_after}  (delta: {n_after - n_before})")
print(f"dropped rows (175.45.176.0~3 & class==Benign): {n_to_drop}")
print(f"residual delete-target rows (should be 0): {leftover_del}")
print("\nCounts for 175.45.176.0~3 after ops:")
print(post_counts_175.to_string() if not post_counts_175.empty else "(none)")
print()

# 厳格チェック
#assert n_src_left == 0, "置換後に 59.166.0.0~3 が残っています。"
#assert leftover_del == 0, "削除対象（175.45.176.0~3 & Benign）が残っています。"

# ========= 3) 保存 =========
df.to_csv(out_csv, index=False)
print(f"Saved to: {out_csv}")