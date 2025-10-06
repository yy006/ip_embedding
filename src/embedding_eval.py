import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# 実験設定の読み込み
DATASET = 'UNSW-NB15'
#EXPERIMENT = '2025-09-30T05-43-47_incremental_ryqsrg6y'
EXPERIMENT = '2025-09-30T05-54-05_single_4vfhlp7f'
json_path = f'experiments/{DATASET}/{EXPERIMENT}/experiment.json'
with open(json_path, 'r') as f:
    config = json.load(f)


# 埋め込みの読み込み
def load_embeddings(path: str | Path):
    p = Path(path)
    with open(p, "rb") as f:
        obj = pickle.load(f)

    print (obj)
    return obj

model = load_embeddings(config['results']['blocks']['001']['model']['model_path'])

def set_labels():
    # embのkeyに存在するIPアドレスの中で、175.45.176.0~3はAttack, それ以外はBenign
    labels = {}
    for ip in model.wv.index_to_key:
        if ip.startswith("175.45.176."):
            labels[ip] = "Attack"
        else:
            labels[ip] = "Benign"
    return labels
labels = set_labels()
#print(labels)

def task_knn():
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    X = []
    y = []
    for ip in model.wv.index_to_key:
        X.append(model.wv[ip])
        y.append(labels[ip])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

task_knn()

# kの値を2~10まで変えて試し、グラフ化する
def task_knn_vary_k():
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    X = []
    y = []
    for ip in model.wv.index_to_key:
        X.append(model.wv[ip])
        y.append(labels[ip])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    accuracies = []
    ks = list(range(2, 11))
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f'k={k}, Accuracy={acc:.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(ks, accuracies, marker='o')
    plt.title('KNN Classifier Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(ks)
    plt.grid()
    plt.show()

import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

def _build_xy(model, labels):
    ips = [ip for ip in model.wv.index_to_key if ip in labels]
    X = np.stack([model.wv[ip] for ip in ips])
    y = np.array([labels[ip] for ip in ips])
    return ips, X, y

def _cosine_neighbors_predict(X, y, k):
    """
    LOOCV で各点を自分以外の上位k近傍の多数決で予測。
    コサイン類似度を使う（= 類似度が高いものが近い）。
    """
    # L2正規化
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-12, None)

    # 類似度行列 (N x N)
    sim = Xn @ Xn.T

    # 自己類似度を -inf にして自分を除外
    np.fill_diagonal(sim, -np.inf)

    # 各行で上位kのインデックス（類似度降順）
    # k が (N-1) を超えないようにする
    N = X.shape[0]
    k_eff = min(k, N-1)
    topk_idx = np.argpartition(-sim, kth=k_eff-1, axis=1)[:, :k_eff]

    # 上位kの中で本当に上位k（厳密順序）に並べ直す
    row_indices = np.arange(N)[:, None]
    topk_sorted = topk_idx[row_indices, np.argsort(-sim[row_indices, topk_idx], axis=1)]

    # 多数決（同点のときは一番近い近傍のラベルでブレーク）
    y_pred = []
    for i in range(N):
        neigh_labels = y[topk_sorted[i]]
        counts = Counter(neigh_labels)
        # 票数最大のラベルたち
        max_votes = max(counts.values())
        cands = [c for c,v in counts.items() if v == max_votes]
        if len(cands) == 1:
            y_pred.append(cands[0])
        else:
            # タイは最も近い（=最も類似度が高い）近傍のラベルを採用
            for nb in topk_sorted[i]:
                if y[nb] in cands:
                    y_pred.append(y[nb])
                    break
    return np.array(y_pred)

def loocv_knn(model, labels, k_list=(1,3,5)):
    """
    すべての点について LOOCV で予測し、kごとに評価。
    Word2Vec向けにコサイン類似度を使用。
    """
    ips, X, y = _build_xy(model, labels)
    classes = np.unique(y)
    results = {}

    for k in k_list:
        y_pred = _cosine_neighbors_predict(X, y, k)
        acc = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=classes)

        print(f"\n=== LOOCV k-NN (k={k}) ===")
        print("Accuracy :", f"{acc:.4f}")
        print("Macro-F1 :", f"{macro_f1:.4f}")
        print("Confusion Matrix (labels order:", list(classes), ")")
        print(cm)
        print(classification_report(y, y_pred, digits=4, zero_division=0))

        results[k] = dict(accuracy=acc, macro_f1=macro_f1, cm=cm, classes=classes)

    # ベストkの表示（macro-F1優先）
    best_k = max(results, key=lambda kk: results[kk]['macro_f1'])
    print(f"\n[Best by Macro-F1] k={best_k}, "
          f"Accuracy={results[best_k]['accuracy']:.4f}, "
          f"Macro-F1={results[best_k]['macro_f1']:.4f}")

    return results

results = loocv_knn(model, labels, k_list=(1,2,3,4,5))
print(results)