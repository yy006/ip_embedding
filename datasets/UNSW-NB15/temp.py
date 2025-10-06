import pandas as pd
# raw_dataやwith_class_nameのデータも後でTimestamp列を追加する

in_csv  = 'ipmap59to175_drop175benign_with_class_name/UNSW-NB15_4_with_class_name_ipmap59to175_drop175benign.csv'

df = pd.read_csv(in_csv, low_memory=False)

# Stime列をTimestamp列に変換
df['Timestamp'] = pd.to_datetime(df['Stime'], unit='s', errors='coerce')

# csv出力
out_csv = 'UNSW-NB15_4_with_class_name_ipmap59to175_drop175benign.csv'
df.to_csv(out_csv, index=False)