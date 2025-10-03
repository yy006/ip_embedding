#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding-based evaluation script for DarkVec / i-DarkVec style pipelines.

Features:
- Load embeddings from .pkl (gensim KeyedVectors / Word2Vec or dict[token]->vector)
- k-NN (cosine) classification with Leave-One-Out CV (LOOCV) or K-Fold
- Linear SVM supervised evaluation (holdout)
- Clustering (HDBSCAN or KMeans) with intrinsic metrics; external metrics if labels present
- Anomaly scoring via LOF
- Reports saved as JSON/CSV

Reference methodology:
- DarkVec: semi-supervised k-NN in embedding space; clustering for unknown-groups discovery
- i-DarkVec: incremental embeddings; same downstream tasks

Author: you :)
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# Optional deps
try:
    from gensim.models import KeyedVectors, Word2Vec  # type: ignore
    _HAS_GENSIM = True
except Exception:
    _HAS_GENSIM = False

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

# HDBSCAN is optional but very useful
try:
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

from sklearn.cluster import KMeans


# ----------------------------
# I/O
# ----------------------------
def load_embeddings(path: str | Path) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Load embeddings from a pickle.
    Accepts:
        - gensim KeyedVectors (model.wv), Word2Vec (uses .wv)
        - dict[str, np.ndarray]
    Returns:
        X: (N, D) vectors
        tokens: list[str] length N
        index: dict token->row_index
    """
    p = Path(path)
    with open(p, "rb") as f:
        obj = pickle.load(f)

    tokens: List[str]
    X: np.ndarray

    # gensim objects
    if _HAS_GENSIM:
        if isinstance(obj, KeyedVectors):
            tokens = list(obj.key_to_index.keys())
            X = obj.get_normed_vectors() if hasattr(obj, "get_normed_vectors") else obj.vectors
        elif isinstance(obj, Word2Vec):
            kv = obj.wv
            tokens = list(kv.key_to_index.keys())
            X = kv.get_normed_vectors() if hasattr(kv, "get_normed_vectors") else kv.vectors
        else:
            # maybe dict?
            if isinstance(obj, dict):
                tokens = list(obj.keys())
                X = np.stack([np.asarray(obj[t], dtype=float) for t in tokens], axis=0)
            else:
                raise TypeError(f"Unsupported pickle type: {type(obj)}")
    else:
        # no gensim; accept dict
        if isinstance(obj, dict):
            tokens = list(obj.keys())
            X = np.stack([np.asarray(obj[t], dtype=float) for t in tokens], axis=0)
        else:
            raise TypeError(
                f"gensim not available; only dict[str, array] is supported. Got: {type(obj)}"
            )

    # ensure float32
    X = np.asarray(X, dtype=np.float32)
    index = {t: i for i, t in enumerate(tokens)}
    return X, tokens, index


def load_labels_csv(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    df = pd.read_csv(path)
    # expected columns: id,label
    expected = {"id", "label"}
    if not expected.issubset(set(map(str.lower, df.columns))):
        # try to normalize case
        cols = {c.lower(): c for c in df.columns}
        if not expected.issubset(cols.keys()):
            raise ValueError("labels CSV must have columns: id,label")
        df = df.rename(columns={cols["id"]: "id", cols["label"]: "label"})
    else:
        # normalize to exact names
        mapping = {c: c.lower() for c in df.columns}
        df = df.rename(columns=mapping)
    return df[["id", "label"]]


def intersect_embeddings_labels(
    tokens: List[str],
    index: Dict[str, int],
    X: np.ndarray,
    labels_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align labels to embeddings by intersection of keys."""
    token_set = set(tokens)
    labels_df = labels_df[labels_df["id"].astype(str).isin(token_set)].copy()
    labels_df = labels_df.dropna(subset=["label"])
    y_labels = labels_df["label"].astype(str).to_numpy()
    row_idx = [index[t] for t in labels_df["id"].astype(str).tolist()]
    X_sub = X[row_idx]
    return X_sub, y_labels, labels_df["id"].astype(str).tolist()


# ----------------------------
# Tasks
# ----------------------------
def task_knn(args):
    X, tokens, index = load_embeddings(args.embeddings)
    labels_df = load_labels_csv(args.labels)
    if labels_df is None:
        raise ValueError("k-NN requires --labels CSV (id,label).")

    Xy, y, ids = intersect_embeddings_labels(tokens, index, X, labels_df)

    # Encode labels to ensure proper stratification
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # LOOCV can be very slow for large N; provide two modes
    if args.cv == "loocv":
        # Implement LOOCV efficiently with KNN by training once on full set?
        # We use a trick: KNN with n_neighbors=k+1 and leave-one-out by masking self.
        # scikit-learn's KNeighborsClassifier doesn't support excluding self directly,
        # so we fallback to manual loop for correctness on cosine distance.
        from sklearn.metrics.pairwise import cosine_distances
        D = cosine_distances(Xy)
        np.fill_diagonal(D, np.inf)  # exclude self
        preds = []
        for i in range(D.shape[0]):
            nn = np.argpartition(D[i], args.k)[:args.k]
            # majority vote
            vals, counts = np.unique(y[nn], return_counts=True)
            preds.append(vals[np.argmax(counts)])
        y_pred = np.array(preds)
    else:
        # Stratified K-Fold CV
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        y_pred = np.empty_like(y, dtype=object)
        for train, test in skf.split(Xy, y_enc):
            clf = KNeighborsClassifier(n_neighbors=args.k, metric="cosine", weights="uniform")
            clf.fit(Xy[train], y[train])
            y_pred[test] = clf.predict(Xy[test])

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=le.classes_).tolist()

    out = {
        "task": "knn",
        "k": args.k,
        "cv": args.cv,
        "folds": args.folds if args.cv == "kfold" else None,
        "n_samples": int(len(y)),
        "n_classes": int(len(le.classes_)),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "labels_order": le.classes_.tolist(),
        "confusion_matrix": cm,
    }
    _write_report(args.report, out)
    print(json.dumps({"accuracy": acc, "f1_macro": f1_macro}, ensure_ascii=False, indent=2))


def task_svm(args):
    X, tokens, index = load_embeddings(args.embeddings)
    labels_df = load_labels_csv(args.labels)
    if labels_df is None:
        raise ValueError("SVM requires --labels CSV (id,label).")

    Xy, y, ids = intersect_embeddings_labels(tokens, index, X, labels_df)

    X_train, X_test, y_train, y_test = train_test_split(
        Xy, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC(C=args.C, dual=False, random_state=args.seed))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    out = {
        "task": "svm",
        "test_size": args.test_size,
        "C": args.C,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    _write_report(args.report, out)
    print(json.dumps({"accuracy": acc, "f1_macro": f1_macro}, ensure_ascii=False, indent=2))


def task_cluster(args):
    X, tokens, index = load_embeddings(args.embeddings)

    # optional label metrics
    labels_df = load_labels_csv(args.labels) if args.labels else None

    # choose algorithm
    if args.algo.lower() == "hdbscan":
        if not _HAS_HDBSCAN:
            raise RuntimeError("HDBSCAN not installed. `pip install hdbscan` or use --algo kmeans")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1))
        # Intrinsic metrics (only on clustered points)
        mask = labels != -1
        intrinsic = {}
        if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
            intrinsic = {
                "silhouette": silhouette_score(X[mask], labels[mask]),
                "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
                "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
            }
        info = {"algo": "hdbscan", "n_clusters": int(n_clusters), "noise": int(noise), **intrinsic}
    elif args.algo.lower() == "kmeans":
        clusterer = KMeans(n_clusters=args.k, n_init="auto", random_state=args.seed)
        labels = clusterer.fit_predict(X)
        info = {
            "algo": "kmeans",
            "n_clusters": int(args.k),
            "silhouette": silhouette_score(X, labels) if len(set(labels)) > 1 else None,
            "calinski_harabasz": calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else None,
            "davies_bouldin": davies_bouldin_score(X, labels) if len(set(labels)) > 1 else None,
        }
    else:
        raise ValueError("--algo must be hdbscan or kmeans")

    out = {"task": "cluster", **info}

    # External metrics if labels provided
    if labels_df is not None:
        # align intersection
        # note: cluster labels are for X order; we must pick labels for tokens in same order
        token_to_label = dict(zip(labels_df["id"].astype(str), labels_df["label"].astype(str)))
        y_ext = np.array([token_to_label.get(t, None) for t in tokens], dtype=object)
        mask = y_ext != None  # noqa: E711
        y_ext = y_ext[mask]
        cl_ext = labels[mask]
        if len(set(y_ext)) > 1 and len(set(cl_ext)) > 1:
            out["external_metrics"] = {
                "adjusted_rand_index": adjusted_rand_score(y_ext, cl_ext),
                "normalized_mutual_info": normalized_mutual_info_score(y_ext, cl_ext),
            }

    _write_report(args.report, out)
    # small console summary
    print(json.dumps(out, ensure_ascii=False, indent=2))


def task_anomaly(args):
    X, tokens, index = load_embeddings(args.embeddings)

    lof = LocalOutlierFactor(n_neighbors=args.k, novelty=False, contamination=args.contamination)
    labels = lof.fit_predict(X)  # -1: outlier, 1: inlier
    scores = -lof.negative_outlier_factor_  # higher -> more anomalous

    df = pd.DataFrame({"id": tokens, "lof_score": scores, "pred": labels})
    out_path = Path(args.report or "anomaly_scores.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("lof_score", ascending=False).to_csv(out_path, index=False)
    print(f"Saved anomaly scores: {out_path}")


# ----------------------------
# Utils
# ----------------------------
def _write_report(path: Optional[str | Path], obj: dict):
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embedding-based evaluation toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # kNN
    sp = sub.add_parser("knn", help="k-NN classification (cosine)")
    sp.add_argument("--embeddings", required=True, help=".pkl path")
    sp.add_argument("--labels", required=True, help="CSV with columns: id,label")
    sp.add_argument("--k", type=int, default=7)
    sp.add_argument("--cv", choices=["loocv", "kfold"], default="loocv")
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--report", default=None, help="output JSON path")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=task_knn)

    # SVM
    sp = sub.add_parser("svm", help="Linear SVM evaluation (holdout)")
    sp.add_argument("--embeddings", required=True)
    sp.add_argument("--labels", required=True)
    sp.add_argument("--test-size", type=float, default=0.2)
    sp.add_argument("--C", type=float, default=1.0)
    sp.add_argument("--report", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=task_svm)

    # Clustering
    sp = sub.add_parser("cluster", help="Clustering in embedding space")
    sp.add_argument("--embeddings", required=True)
    sp.add_argument("--labels", default=None, help="optional CSV for external metrics")
    sp.add_argument("--algo", choices=["hdbscan", "kmeans"], default="hdbscan")
    sp.add_argument("--min-cluster-size", type=int, default=30)
    sp.add_argument("--k", type=int, default=10, help="kmeans clusters if --algo kmeans")
    sp.add_argument("--report", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=task_cluster)

    # Anomaly
    sp = sub.add_parser("anomaly", help="LOF anomaly scoring")
    sp.add_argument("--embeddings", required=True)
    sp.add_argument("--k", type=int, default=20)
    sp.add_argument("--contamination", type=float, default="auto")
    sp.add_argument("--report", default="anomaly_scores.csv")
    sp.set_defaults(func=task_anomaly)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
