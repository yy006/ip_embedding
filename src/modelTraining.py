from config import *
from preprocess import load_raw_data, filter_data, get_next_day
from corpus import get_corpus
from word2vec import Word2Vec
from torch_word2vec import TorchWord2Vec
import numpy as np
import time
from logger import ExperimentLogger, save_model_and_dict, corpus_basic_stats

from pathlib import Path
from copy import deepcopy
import csv

SAVE = True

# === ベースのパラメータ（元の params[0] と同じ） ===
BASE_PARAMS = {
    'corpus': {
        'services': 'auto',
        'without_duplicates': True,
        'top_ports': 300,
    },
    'word2vec': {
        'c': 25,
        'e': 50,
        'epochs': 2,
        'method': 'incremental',
        'alpha_anom': 0.5,   # デフォルトα（スイープしないときに使う）
    },
}

print("CONFIG TRAINING_MODE:", TRAINING_MODE)
print("BLOCKS:", BLOCKS)

artifact_root = ARTIFACTS_ROOT

# =========================
# スイープ設定
# =========================

# True にすると α をスイープ
DO_ALPHA_SWEEP = True
ALPHAS = [0, 0, 0, 0, 0, 0]  # ここに試したい alpha のリスト

# True にすると single / incremental 両方回す
DO_MODE_SWEEP = True
RUN_MODES = ["incremental"]  # 必要なら ["incremental"] などに変更

R_list = [None, 0.5, 1.0, 2.0, 10.0]  # ノルム制約の候補リスト

# mappingのid生成
rand8 = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=8))

# alpha, mode, attack, run_id を記録するハッシュ化されたidを持つCSVのパス 
ALPHA_MAPPING_PATH = Path(artifact_root) / ("alpha_sweep_mapping_" + rand8 + ".csv")



def append_alpha_mapping(mapping_path: Path,
                         alpha: float,
                         mode: str,
                         run_id: str,
                         radius: float | None = None
                         ):
    """alpha と (mode, block_id, モデルファイル) の対応を CSV に追記"""
    mapping_path = Path(mapping_path)
    is_new = not mapping_path.exists()

    with mapping_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["alpha_anom", "mode", "block_id", "Radius"])
        writer.writerow([alpha, mode, run_id, radius])


def run_training(mode: str,
                 params: dict,
                 alpha: float | None = None,
                 mapping_path: Path | None = None,
                 radius: float | None = None
                 ):
    """
    1回分の学習を実行する。
    - mode: "single" or "incremental"
    - params: {'corpus': {...}, 'word2vec': {...}}
    - alpha: ログ出力・対応表用（None の場合もOK）
    """
    # === 実験ロガー初期化 ===
    exp_logger = ExperimentLogger(
        artifact_root=artifact_root,
        dataset=DATASET,
        mode=mode,      # ここを TRAINING_MODE ではなく引数の mode に
        blocks=BLOCKS,
        params=params,
    )

    alpha_info = f"[alpha={alpha}] " if alpha is not None else ""

    if mode == "single":
        block_id = 1
        t_block_start = time.perf_counter()

        exp_logger.block_start(block_id)

        # 元コードそのまま
        raw_data = load_raw_data(len(BLOCKS))
        filtered = filter_data(raw_data, BLOCKS, len(BLOCKS))
        ips_seqs, label_seqs = get_corpus(filtered, **params['corpus'])
        corpus_stats = corpus_basic_stats(ips_seqs)

        model = TorchWord2Vec(**params['word2vec'])
        t_train_start = time.perf_counter()
        model.train_with_labels(
                ips_seqs,
                label_seqs,
                batch_size=1024,
                min_count=0,
                save=SAVE,
            )
        t_train_end = time.perf_counter()

        # モデル・辞書保存
        mpath = exp_logger.model_path(block_id)
        dpath = exp_logger.dict_path(block_id)
        save_model_and_dict(model, {}, mpath, dpath)

        vocab_size = len(model.model.wv.index_to_key)
        vector_size = getattr(model.model.wv, "vector_size", None) or getattr(model.model, "vector_size", 0)

        t_block_end = time.perf_counter()
        print(
            f"{alpha_info}[mode=single] "
            f"Block {block_id:03d} training time: {t_train_end - t_train_start:.2f} sec, "
            f"total time: {t_block_end - t_block_start:.2f} sec"
        )

        exp_logger.block_end(
            block_id,
            vocab_size=vocab_size,
            vector_size=vector_size,
            corpus_stats=corpus_stats,
        )
        exp_logger.finalize()

        # αスイープ中なら対応表に記録
        if mapping_path is not None and alpha is not None:
            append_alpha_mapping(mapping_path, alpha, mode, exp_logger.run_id)

        print("wv:", model.model.wv)

    elif mode == "incremental":
        for block_id in BLOCKS.keys():
            t_block_start = time.perf_counter()

            exp_logger.block_start(block_id)

            # 元コードそのまま
            raw_data = load_raw_data(block_id)
            filtered = filter_data(raw_data, BLOCKS, block_id)
            ips_seqs, label_seqs = get_corpus(filtered, **params['corpus'])
            corpus_stats = corpus_basic_stats(ips_seqs)

            model = TorchWord2Vec(**params['word2vec'])

            t_train_start = time.perf_counter()
            model.train_with_labels(
                ips_seqs,
                label_seqs,
                batch_size=1024,
                min_count=0,
                save=SAVE,
            )
            t_train_end = time.perf_counter()

            # モデル・辞書保存
            mpath = exp_logger.model_path(block_id)
            dpath = exp_logger.dict_path(block_id)
            save_model_and_dict(model, {}, mpath, dpath)

            vocab_size = len(model.model.wv.index_to_key)
            vector_size = getattr(model.model.wv, "vector_size", None) or getattr(model.model, "vector_size", 0)

            t_block_end = time.perf_counter()
            print(
                f"{alpha_info}[mode=incremental] "
                f"Block {block_id:03d} training time: {t_train_end - t_train_start:.2f} sec, "
                f"total time: {t_block_end - t_block_start:.2f} sec"
            )

            exp_logger.block_end(
                block_id,
                vocab_size=vocab_size,
                vector_size=vector_size,
                corpus_stats=corpus_stats,
            )
            exp_logger.finalize()

            # αスイープ中なら対応表に記録 各実行につき一度だけ
            if mapping_path is not None and alpha is not None and block_id == 1:
                append_alpha_mapping(mapping_path, alpha, mode, exp_logger.run_id, radius)

            print("wv:", model.model.wv)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ======================================
# メイン：スイープあり / なしを切り替え
# ======================================
if __name__ == "__main__":
    # 「スイープしない」場合:
    # DO_ALPHA_SWEEP = False
    # DO_MODE_SWEEP  = False
    # としておけば、ほぼ元の挙動に戻る

    if not DO_ALPHA_SWEEP and not DO_MODE_SWEEP:
        # === 完全に「元と同じ」動き：config の TRAINING_MODE と BASE_PARAMS 1回分だけ ===
        params = deepcopy(BASE_PARAMS)

        # alpha は params の値をそのまま使う
        alpha = params['word2vec'].get('alpha_anom', None)
        run_training(TRAINING_MODE, params, alpha=alpha, mapping_path=None)

    else:
        # === αスイープ × モードスイープ ===
        # モードを決める
        modes = RUN_MODES if DO_MODE_SWEEP else [TRAINING_MODE]

        # alpha の候補を決める
        if DO_ALPHA_SWEEP:
            alpha_list = ALPHAS
        else:
            alpha_list = [BASE_PARAMS['word2vec']['alpha_anom']]

        for mode in modes:
            for alpha in alpha_list:
                print(f"\n===== mode={mode}, alpha={alpha} =====")

                params = deepcopy(BASE_PARAMS)
                params['word2vec']['alpha_anom'] = alpha

                for R in R_list:
                    print(f"\n----- norm_radius={R} -----")
                    if R is not None:
                        params['word2vec']['norm_radius'] = R
                    else:
                        if 'norm_radius' in params['word2vec']:
                            del params['word2vec']['norm_radius']

                    run_training(
                        mode=mode,
                        params=params,
                        alpha=alpha,
                        mapping_path=ALPHA_MAPPING_PATH if DO_ALPHA_SWEEP else None,
                        radius=R
                    )
