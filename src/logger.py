# logger.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json, time, random, string
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config import *
from word2vec import Word2Vec
import numpy as np

ISO_FMT = "%Y-%m-%dT%H-%M-%S"
TZ_JST = ZoneInfo("Asia/Tokyo")

def _now_iso() -> str:
    # ローカル時刻で保存（必要なら timezone.utc に）
    return datetime.now(TZ_JST).strftime(ISO_FMT)

def _rand8() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

class ExperimentLogger:
    """
    実験ログを JSON で安全に書き出す薄いラッパー。
    - JSONは run_dir/experiment.json に保存
    - ブロックごとに append 的に更新
    - モデル/辞書の保存先パスを規約で生成
    """
    def __init__(
        self,
        artifact_root: Path,          # 例: ROOT/'experiments'
        dataset: str,                 # 例: DATASET
        mode: str,                    # "single" | "incremental"
        blocks: Dict[int, Path],# 例: BLOCKS（int->path でも str->path でもOK）
        params: Dict[str, Any],       # 例: params[0]
        run_id: Optional[str] = None, # 指定なければ自動生成
        schema_version: str = "1.0",
    ):
        started_at = _now_iso()
        rid = run_id or f"{started_at}_{mode}_{_rand8()}"
        self.run_id = rid
        self.dataset = dataset
        self.mode = mode
        self.params = params

        # アーティファクトのルート: <artifact_root>/<dataset>/<run_id>/
        self.run_dir = artifact_root / dataset / rid
        self.models_dir = self.run_dir / "models"
        self.dicts_dir  = self.run_dir / "dicts"
        self.logs_path  = self.run_dir / "experiment.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.dicts_dir.mkdir(parents=True, exist_ok=True)

        # BLOCKSを "1": "path" の形に正規化
        blocks_norm: Dict[str, str] = {}
        for k, v in blocks.items():
            ks = f"{int(k)}" if isinstance(k, (int,)) or str(k).isdigit() else str(k)
            blocks_norm[ks] = str(v)

        self.state: Dict[str, Any] = {
            "schema_version": schema_version,
            "run_id": rid,
            "dataset": dataset,
            "mode": mode,
            "started_at": started_at,
            "blocks": blocks_norm,
            "params": params,
            "results": {
                "overall": {},
                "blocks": {}  # "001": {...}, "002": {...}
            }
        }
        _atomic_write(self.logs_path, self.state)

        # 実行時間集計
        self._t0 = time.perf_counter()
        self._block_t0: Optional[float] = None

    # ========= パス規約 =========
    def model_path(self, block_id: int) -> Path:
        return self.models_dir / f"model_block_{block_id:03d}"

    def dict_path(self, block_id: int) -> Path:
        return self.dicts_dir / f"dict_block_{block_id:03d}.json"

    # ========= JSON 更新系 =========
    def block_start(self, block_id: int) -> None:
        self._block_t0 = time.perf_counter()

    def block_end(
        self,
        block_id: int,
        vocab_size: int,
        vector_size: int,
        corpus_stats: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert self._block_t0 is not None, "block_start() を先に呼んでください"
        elapsed = time.perf_counter() - self._block_t0
        key = f"{block_id:03d}"
        b: Dict[str, Any] = {
            "block_id": block_id,
            "wall_time_sec": round(elapsed, 3),
            "model": {
                "vocab_size": vocab_size,
                "vector_size": vector_size,
                "model_path": str(self.model_path(block_id)),
                "dict_path":  str(self.dict_path(block_id)),
            },
            "corpus_stats": corpus_stats,
        }
        if extra:
            b.update(extra)

        self.state["results"]["blocks"][key] = b
        _atomic_write(self.logs_path, self.state)
        self._block_t0 = None

    def finalize(self) -> None:
        total = time.perf_counter() - self._t0
        self.state["results"]["overall"]["total_wall_time_sec"] = round(total, 3)
        self.state["finished_at"] = _now_iso()
        _atomic_write(self.logs_path, self.state)

def save_model_and_dict(model: Word2Vec, dict_obj: dict, model_path: Path, dict_path: Path):
    """
    Word2Vec 実装差異を吸収しつつ保存。存在するメソッドだけ呼ぶ。
    """
    model.model.save(str(model_path))

def corpus_basic_stats(corpus):
    # corpus: List[List[token]]
    doc_lens = [len(doc) for doc in corpus]
    uniq_lens = [len(set(doc)) for doc in corpus]
    if not doc_lens:
        return {
            "ndocs": 0,
            "min_words": 0, "avg_words": 0, "max_words": 0,
            "min_unique_words": 0, "avg_unique_words": 0, "max_unique_words": 0,
        }
    return {
        "ndocs": int(len(corpus)),
        "min_words": int(np.min(doc_lens)),
        "avg_words": float(np.mean(doc_lens)),
        "max_words": int(np.max(doc_lens)),
        "min_unique_words": int(np.min(uniq_lens)),
        "avg_unique_words": float(np.mean(uniq_lens)),
        "max_unique_words": int(np.max(uniq_lens)),
    }
