import numpy as np

def metrics_from_confusion(cm):
    """
    cm: [[TN, FP], [FN, TP]] の2x2混合行列（list or np.ndarray）
    戻り値: 指標の辞書
    """
    cm = np.asarray(cm, dtype=float)
    assert cm.shape == (2, 2), "cm must be 2x2 as [[TN, FP], [FN, TP]]"
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    # 基本
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0  # TPR / Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # TNR
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / cm.sum() if cm.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_acc = (recall + specificity) / 2.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
    }

# 使用例（あなたが提示した混合行列）
cm = [
    [
      26365,
      4744
    ],
    [
      1242,
      4938
    ]
]
m = metrics_from_confusion(cm)

# 見やすく表示
for k in ["precision", "recall", "f1", "accuracy", "specificity", "fpr", "fnr", "balanced_accuracy"]:
    print(f"{k:>18}: {m[k]:.4f}")
