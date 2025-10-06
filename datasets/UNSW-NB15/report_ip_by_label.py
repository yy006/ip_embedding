import pandas as pd

# 入力
#in_csv  = "ipmap59to175_drop175benign_with_class_name/UNSW-NB15_1_with_class_name_ipmap59to175_drop175benign.csv"
in_csv = 'by2h/datasetX_2015021802_2015021804_by2h.csv'
out_csv = "2015021802_2015021804_by2h_report_ip_by_label.csv"

#ip_col    = "Src IP Addr"
#label_col = "class"

ip_col    = "srcip"
label_col = "attack_cat"

df = pd.read_csv(in_csv, low_memory=False)
df.columns = df.columns.str.strip()
if ip_col not in df.columns or label_col not in df.columns:
    raise KeyError(f"列が見つかりません: {ip_col}, {label_col}")

df = df.dropna(subset=[ip_col, label_col])

# ip×label の件数 → ワイド形式（行=ip, 列=label, 値=count）
wide_df = (
    df.groupby([ip_col, label_col]).size().rename("count").reset_index()
      .pivot(index=ip_col, columns=label_col, values="count")
      .fillna(0).astype("int64").sort_index()
)

wide_df.to_csv(out_csv)
print("Saved:", out_csv)
print(wide_df.head())