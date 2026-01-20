def run_time_split():
    # リソース使用量を表示
    print_system_resource_usage()
    print_gpu_resource_usage()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = "UNSW_NB15" # CTU13 or CICIDS2017 or CIDDS001 or UNSW_NB15
    time_processed_df = data_reader(dataset) # CTU13 or CICIDS2017 or CIDDS001
 
    # 時間を割り振る
    if dataset == "CTU13":
        time_processed_df = utils.assign_time_number(time_processed_df, 'CTU13', 'N_splits', N=6)

    # 実験用にデータを改変する
    if True:
        time_processed_df = transformDatasetForTesting(time_processed_df, dataset, "replace_IP")

    # time列がkのみを残して学習
    if False:
        time_processed_df = time_processed_df[time_processed_df["time"] == 1]
        print(0)
    print(time_processed_df)

    # データ指定した数で分割して、k番目を取ってくる
    if False:
        time_processed_df = get_kth_interval(time_processed_df, 200000, 0) # インデックスは0から始まる

    # ランダムにdfからk個の行を選択
    if False:
        time_processed_df = time_processed_df.sample(7500, random_state=1)

    # 全区間でシャッフルする
    if True:
        time_processed_df = time_processed_df.sample(frac=1, random_state=1).reset_index(drop=True)


    # データの統計情報を取得
    label_counts, total_data_count, unique_data_types, attack_ips, ips_atk_and_benign = analyze_data(time_processed_df, dataset=dataset)

    # 統計情報の出力
    print("各ラベルごとのデータ数:\n", label_counts)
    print("総データ数:", total_data_count)
    print("データの種類:", unique_data_types)
    print("攻撃ラベルを持つIPの攻撃出現回数:\n", attack_ips)
    print("攻撃ラベルを持つIPの攻撃と通常の出現回数:\n", ips_atk_and_benign)

    result = analyze_timestamps(time_processed_df)
    print(result)
    
    X = time_processed_df.iloc[:, :5] # 文脈には5列だけ使う
    print(X)
    d = X.to_numpy()
    w2v,v2w = preprocess._w2v(d)
    corpus = pd.DataFrame(preprocess._corpus(d, w2v)).to_numpy()
    #print(corpus)
    freq  = preprocess._frequency(d)
    #print(freq)
    
    batch_size = 64 #1024
    train = preprocess._data_loader(corpus, batch_size)
    #print(train)

    model = trainer.Trainer(w2v,v2w,freq,emb_dim=32)
    model.fit(data = train,max_epoch=10,batch_size=batch_size,neg_num=10, patience_limit=3)

    # モデルの状態辞書を取得
    # model_state = model.model.state_dict()
    

if __name__ == '__main__':
    run_time_split()