# src_ip2vec/run.py
from __future__ import annotations

from .config import *

# src_ip2vec 配下（作り直す前提）
from .preprocess import load_raw_data, filter_data  # あなたの既存IFに合わせて用意
from .trainer import TorchIP2Vec                    # 先に提示したSGNS trainer（または同等）
from .logger import ExperimentLogger, save_model_and_dict
from .vocab import build_vocab_from_df_ip2vec        # DF→token2id/id2token/freqs

import numpy as np
import time
import csv
from pathlib import Path
from copy import deepcopy

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

SAVE = True

# =====================================================
# ベースパラメータ（IP2Vec用に寄せる）
# =====================================================
BASE_PARAMS = {
    "ip2vec": {
        # preprocess/filter 側で吸収してもOKだが、列名をここで明示できるようにしておく
        "src_col": "srcip",
        "dst_col": "dstip",
        "dport_col": "dsport",
        "proto_col": "proto",
        "use_prefix": True,     # "SRCIP:" など prefix を付ける
    },
    "word2vec": {  # SGNS trainer 用（TorchIP2Vec側）
        "e": 50,
        "epochs": 2,
        "neg_k": 5,
        "lr": 0.025,
        "norm_radius": None,    # 必要なら sweep
    },
}

artifact_root = ARTIFACTS_ROOT

# =====================================================
# スイープ設定（枠は維持）
# =====================================================
DO_ALPHA_SWEEP = True
ALPHAS = [0]  # IP2Vec側でalphaを使わないなら 0 固定でOK（mapping用途のみ残す）

DO_MODE_SWEEP = True
RUN_MODES = ["incremental", "single"]

R_list = [None]
normal_pull_lambda_list = [0]  # IP2Vec側で未使用でもmapping/ログ用に残す

Note = "IP2Vec run"

DO_ATTACK_SWEEP = False
ATTACK_LIST = ["Fuzzers", "Reconnaissance", "DoS", "Exploits", "Shellcode"]

rand8 = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=8))
ALPHA_MAPPING_PATH = Path(artifact_root) / (f"alpha_sweep_mapping_{rand8}.csv")


# =====================================================
# BLOCKS 構築（あなたの元コードを踏襲）
# =====================================================
def build_blocks_for_attack(attack: str) -> dict[int, Path]:
    data_root = ROOT / "datasets" / DATASET
    data_path = data_root / PREPROCESS / attack
    #data_path = data_root / PREPROCESS

    blocks: dict[int, Path] = {
        1: data_path / f"2015012218_2015012220_by2h_{attack}.csv",
        2: data_path / f"2015012220_2015012222_by2h_{attack}.csv",
        3: data_path / f"2015012222_2015012300_by2h_{attack}.csv",
        4: data_path / f"2015012300_2015012302_by2h_{attack}.csv",
        5: data_path / f"2015021800_2015021802_by2h_{attack}.csv",
        6: data_path / f"2015021802_2015021804_by2h_{attack}.csv",
    }

    """
    blocks: dict[int, Path] = {
        1: data_path / "modified_Monday-WorkingHours.pcap_ISCX.csv",
        2: data_path / "modified_Monday-WorkingHours.pcap_ISCX.csv",
        3: data_path / "output_filtered.csv",
    }
    """
    return blocks


# =====================================================
# mapping CSV
# =====================================================
def append_alpha_mapping(
    mapping_path: Path,
    alpha: float,
    mode: str,
    attack: str,
    run_id: str,
    radius: float | None = None,
    normal_pull_lambda: float | None = None,
):
    mapping_path = Path(mapping_path)
    is_new = not mapping_path.exists()

    with mapping_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["alpha_anom", "mode", "attack", "run_id", "Radius", "normal_pull_lambda"])
        writer.writerow([alpha, mode, attack, run_id, radius, normal_pull_lambda])


# =====================================================
# 追加：incremental 用の vocab 拡張 + unigram 更新 + embedding拡張
# =====================================================
def _tokenize_row_ip2vec(r, *, src_col, dst_col, dport_col, proto_col, use_prefix: bool):
    # vocab.py と同じトークン化ルールに必ず合わせる
    def tok(kind, v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):  # NaN
            return None
        if kind == "DPORT":
            try:
                v = int(v)
            except Exception:
                pass
        return f"{kind}:{v}" if use_prefix else str(v)

    return [
        tok("SRCIP", r.get(src_col)),
        tok("DSTIP", r.get(dst_col)),
        tok("DPORT", r.get(dport_col)),
        tok("PROTO", r.get(proto_col)),
    ]


def expand_vocab_and_model_ip2vec(
    trainer: TorchIP2Vec,
    df_new,
    *,
    src_col,
    dst_col,
    dport_col,
    proto_col,
    use_prefix: bool,
    min_count: int = 0,
):
    """
    - 新ブロック df_new を見て token2id/id2token を拡張
    - unigram_table を更新（全語彙の頻度に基づく）
    - in/out embedding を行追加して拡張（初期化: in=uniform, out=zeros）
    """
    # trainer に頻度を持たせる（無い場合は初回構築）
    # ここでは "累積頻度" を辞書で保持する方針
    if not hasattr(trainer, "_freq_counter") or trainer._freq_counter is None:
        trainer._freq_counter = {}

    freq = trainer._freq_counter

    # 新ブロック分の頻度を追加
    for _, r in df_new.iterrows():
        toks = _tokenize_row_ip2vec(
            r,
            src_col=src_col, dst_col=dst_col, dport_col=dport_col, proto_col=proto_col,
            use_prefix=use_prefix,
        )
        for t in toks:
            if t is None:
                continue
            freq[t] = freq.get(t, 0) + 1

    # min_count を満たす語彙を再構築（IDは安定させたいので「追加のみ」を基本にする）
    # ただし、すでに token2id にあるものは残す
    new_tokens = []
    for t, c in freq.items():
        if c >= min_count and t not in trainer.token2id:
            new_tokens.append(t)
    new_tokens.sort()

    if not new_tokens:
        # unigramだけ更新して終了
        _refresh_unigram_from_freq(trainer, freq)
        return

    old_vocab_size = len(trainer.token2id)

    # token2id/id2token 拡張
    for t in new_tokens:
        trainer.token2id[t] = len(trainer.token2id)
    trainer.id2token = {i: t for t, i in trainer.token2id.items()}

    new_vocab_size = len(trainer.token2id)

    # unigram 更新
    _refresh_unigram_from_freq(trainer, freq)

    # embedding テーブル拡張
    _expand_embeddings(trainer, old_vocab_size, new_vocab_size)


def _refresh_unigram_from_freq(trainer: TorchIP2Vec, freq: dict[str, int]):
    # unigram^(3/4)
    import numpy as _np
    tokens = sorted(trainer.token2id.keys(), key=lambda x: trainer.token2id[x])
    counts = _np.array([freq.get(t, 1) for t in tokens], dtype=_np.float64)  # 1で下駄
    probs = counts ** 0.75
    probs = probs / probs.sum()
    trainer.unigram_table = torch.tensor(probs, dtype=torch.float32, device=trainer.device)


def _expand_embeddings(trainer: TorchIP2Vec, old_vocab_size: int, new_vocab_size: int):
    # trainer.model.in_embed / out_embed を拡張
    model = trainer.model
    assert model is not None

    D = model.in_embed.weight.shape[1]
    device = trainer.device

    old_in = model.in_embed.weight.data
    old_out = model.out_embed.weight.data

    # 新しい Embedding を作り直してコピー（nn.Embeddingはresizeが面倒なので）
    new_in = torch.nn.Embedding(new_vocab_size, D).to(device)
    new_out = torch.nn.Embedding(new_vocab_size, D).to(device)

    # 初期化（word2vec風）
    initrange = 0.5 / D
    torch.nn.init.uniform_(new_in.weight, -initrange, initrange)
    torch.nn.init.zeros_(new_out.weight)

    # 既存分をコピー
    new_in.weight.data[:old_vocab_size].copy_(old_in)
    new_out.weight.data[:old_vocab_size].copy_(old_out)

    model.in_embed = new_in
    model.out_embed = new_out


# =====================================================
# 1回分の学習
# =====================================================
def run_training(
    mode: str,
    params: dict,
    blocks: dict[int, Path],
    attack: str | None = None,
    alpha: float | None = None,
    mapping_path: Path | None = None,
    radius: float | None = None,
    normal_pull_lambda: float | None = None,
    note: str | None = None,
):
    # Logger
    exp_logger = ExperimentLogger(
        artifact_root=artifact_root,
        dataset=DATASET,
        mode=mode,
        blocks=blocks,
        params=params,
        note=note,
    )

    alpha_info = f"[alpha={alpha}] " if alpha is not None else ""
    attack_info = f"[attack={attack}] " if attack is not None else ""
    info_prefix = attack_info + alpha_info + f"[mode={mode}] "

    ip2 = params["ip2vec"]
    w2v = params["word2vec"]

    # norm_radius sweep
    if radius is not None:
        w2v["norm_radius"] = radius
    else:
        w2v.pop("norm_radius", None)


    if mode == "single":
        block_id = 1
        exp_logger.block_start(block_id)

        raw = load_raw_data(blocks, len(blocks), mode="single")
        df = filter_data(raw, blocks, len(blocks), mode="single")

        trainer = TorchIP2Vec(**w2v)

        t_train_start = time.perf_counter()
        trainer.train_ip2vec(
            df,
            batch_size=1024,
            min_count=0,
            incremental=False,   # ★ trainerに任せる
            src_col=ip2["src_col"], dst_col=ip2["dst_col"],
            dport_col=ip2["dport_col"], proto_col=ip2["proto_col"],
            use_prefix=ip2["use_prefix"],
        )
        t_train_end = time.perf_counter()

        mpath = exp_logger.model_path(block_id)
        dpath = exp_logger.dict_path(block_id)
        save_model_and_dict(trainer, trainer.token2id, mpath, dpath)

        exp_logger.block_end(
            block_id,
            vocab_size=len(trainer.token2id),
            vector_size=w2v["e"],
            corpus_stats=None,
        )
        exp_logger.finalize()

        # mapping CSV（そのまま）

    elif mode == "incremental":
        trainer = None

        for block_id in blocks.keys():
            exp_logger.block_start(block_id)

            raw = load_raw_data(blocks, block_id, mode="incremental")
            df = filter_data(raw, blocks, block_id, mode="incremental")

            if block_id == 1:
                trainer = TorchIP2Vec(**w2v)

                t_train_start = time.perf_counter()
                trainer.train_ip2vec(
                    df,
                    batch_size=1024,
                    min_count=0,
                    incremental=False,  # ★ 初回は trainer に作らせる
                    src_col=ip2["src_col"], dst_col=ip2["dst_col"],
                    dport_col=ip2["dport_col"], proto_col=ip2["proto_col"],
                    use_prefix=ip2["use_prefix"],
                )
                t_train_end = time.perf_counter()

                # ★ freq を持たせたいならここで初期化（後述）

            else:
                assert trainer is not None
                expand_vocab_and_model_ip2vec(
                    trainer,
                    df,
                    src_col=ip2["src_col"], dst_col=ip2["dst_col"],
                    dport_col=ip2["dport_col"], proto_col=ip2["proto_col"],
                    use_prefix=ip2["use_prefix"],
                    min_count=0,
                )

                t_train_start = time.perf_counter()
                trainer.train_ip2vec(
                    df,
                    batch_size=1024,
                    min_count=0,
                    incremental=True,
                    src_col=ip2["src_col"], dst_col=ip2["dst_col"],
                    dport_col=ip2["dport_col"], proto_col=ip2["proto_col"],
                    use_prefix=ip2["use_prefix"],
                )
                t_train_end = time.perf_counter()

            mpath = exp_logger.model_path(block_id)
            dpath = exp_logger.dict_path(block_id)
            save_model_and_dict(trainer, trainer.token2id, mpath, dpath)

            exp_logger.block_end(
                block_id,
                vocab_size=len(trainer.token2id),
                vector_size=w2v["e"],
                corpus_stats=None,
            )

        exp_logger.finalize()

        if mapping_path is not None and alpha is not None and attack is not None:
            append_alpha_mapping(mapping_path, alpha, mode, attack, exp_logger.run_id, radius, normal_pull_lambda)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =====================================================
# main：attack / alpha / mode sweep（枠は踏襲）
# =====================================================
if __name__ == "__main__":
    attacks = ATTACK_LIST if DO_ATTACK_SWEEP else [ATTACK]
    modes = RUN_MODES if DO_MODE_SWEEP else [TRAINING_MODE]
    alpha_list = ALPHAS if DO_ALPHA_SWEEP else [0]

    REPEAT = 5

    for attack in attacks:
        blocks = build_blocks_for_attack(attack)
        print("\n==============================")
        print(f"ATTACK = {attack}")
        print("BLOCKS:")
        for k, v in blocks.items():
            print(f"  {k}: {v}")
        print("==============================")

        for mode in modes:
            for alpha in alpha_list:
                for repeat_id in range(REPEAT):
                    print(f"\n===== attack={attack}, mode={mode}, alpha={alpha}, repeat={repeat_id} =====")

                    params = deepcopy(BASE_PARAMS)

                    for R in R_list:
                        print(f"\n----- norm_radius={R} -----")
                        # norm_radius は word2vec セクションで処理（run_training内で反映）

                        for normal_pull_lambda in normal_pull_lambda_list:
                            print(f"\n----- normal_pull_lambda={normal_pull_lambda} -----")

                            run_training(
                                mode=mode,
                                params=params,
                                blocks=blocks,
                                attack=attack,
                                alpha=alpha,
                                mapping_path=ALPHA_MAPPING_PATH if DO_ALPHA_SWEEP else None,
                                radius=R,
                                normal_pull_lambda=normal_pull_lambda,
                                note=Note,
                            )
