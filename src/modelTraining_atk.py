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
        'e': 12,
        'epochs': 2,
        'method': 'incremental',
        'alpha_anom': 0.5,   # デフォルトα（スイープしないときに使う）
    },
}

print("CONFIG TRAINING_MODE:", TRAINING_MODE)
#print("CONFIG ATTACK (from config.py):", ATTACK)

artifact_root = ARTIFACTS_ROOT

# =====================================================
# スイープ設定
# =====================================================

# True にすると α をスイープ
DO_ALPHA_SWEEP = True
#ALPHAS = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
ALPHAS = [0]

# True にすると single / incremental 両方回す
DO_MODE_SWEEP = True
RUN_MODES = ["incremental"]  # 片方だけ試したいときはここを編集

R_list = [None, 0.5 ,1.0, 5.0]  # ノルム制約の候補リスト
#R_list = [None]

normal_pull_lambda_list = [0]  # 正常点引き寄せ項のλ候補リスト

# True にすると攻撃ラベルもスイープ
DO_ATTACK_SWEEP = True

ATTACK_LIST = [
    "Generic",
    "DoS",
    "Analysis",
    "Backdoor",
    "Exploits",
    "Fuzzers",
    "Worms",
    "Reconnaissance",
    "Shellcode",
]

# mappingのid生成
rand8 = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=8))

# alpha, mode, attack, run_id を記録するハッシュ化されたidを持つCSVのパス 
ALPHA_MAPPING_PATH = Path(artifact_root) / ("alpha_sweep_mapping_" + rand8 + ".csv")


# =====================================================
# ATTACK ごとの BLOCKS を作るヘルパー
# =====================================================

def build_blocks_for_attack(attack: str) -> dict[int, Path]:
    """
    config.py の書き方と同じルールで、
    指定された attack 用の BLOCKS を作る。
    """
    data_root = ROOT / "datasets" / DATASET
    data_path = data_root / PREPROCESS / attack

    blocks: dict[int, Path] = {
        1: data_path / f"2015012218_2015012220_by2h_{attack}.csv",
        2: data_path / f"2015012220_2015012222_by2h_{attack}.csv",
        3: data_path / f"2015012222_2015012300_by2h_{attack}.csv",
        4: data_path / f"2015012300_2015012302_by2h_{attack}.csv",
        5: data_path / f"2015021800_2015021802_by2h_{attack}.csv",
        6: data_path / f"2015021802_2015021804_by2h_{attack}.csv",
    }
    return blocks


# =====================================================
# α–mode–attack 対応表 CSV
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
    """
    alpha, mode, attack, run_id の対応を CSV に追記。
    後の評価スクリプトで使うことを想定。
    """
    mapping_path = Path(mapping_path)
    is_new = not mapping_path.exists()

    with mapping_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            # attack も含めるが、eval側は alpha_anom/mode/run_id だけ見てもOK
            writer.writerow(["alpha_anom", "mode", "attack", "run_id", "Radius", "normal_pull_lambda"])
        writer.writerow([alpha, mode, attack, run_id, radius, normal_pull_lambda])


# =====================================================
# 1回分の学習を実行する関数
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

):
    """
    1回分の学習を実行する。

    - mode: "single" or "incremental"
    - params: {'corpus': {...}, 'word2vec': {...}}
    - blocks: {block_id: Path} （この run で使うブロック定義）
    - attack: 攻撃ラベル（ログ・CSV用）
    - alpha: alpha_anom（ログ・CSV用）
    - mapping_path: 対応表 CSV のパス（None のときは書かない）
    """

    # === 実験ロガー初期化 ===
    exp_logger = ExperimentLogger(
        artifact_root=artifact_root,
        dataset=DATASET,
        mode=mode,
        blocks=blocks,
        params=params,
    )

    alpha_info = f"[alpha={alpha}] " if alpha is not None else ""
    attack_info = f"[attack={attack}] " if attack is not None else ""
    info_prefix = attack_info + alpha_info + f"[mode={mode}] "

    if mode == "single":
        # もともと block_id=1〜len(BLOCKS) をロードして学習、という設計なら
        # そのまま len(blocks) を渡す
        block_id = 1
        t_block_start = time.perf_counter()

        exp_logger.block_start(block_id)

        raw_data = load_raw_data(len(blocks))
        filtered = filter_data(raw_data, blocks, len(blocks))
        ips_seqs = get_corpus(filtered, **params['corpus'])
        corpus_stats = corpus_basic_stats(ips_seqs)

        model = TorchWord2Vec(**params['word2vec'])

        t_train_start = time.perf_counter()
        model.train(
            ips_seqs,
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
        vector_size = getattr(model.model.wv, "vector_size", None) or getattr(
            model.model, "vector_size", 0
        )

        t_block_end = time.perf_counter()
        print(
            f"{info_prefix}"
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
        if mapping_path is not None and alpha is not None and attack is not None:
            append_alpha_mapping(mapping_path, alpha, mode, attack, exp_logger.run_id, radius, normal_pull_lambda)

        print("wv:", model.model.wv)

    elif mode == "incremental":
        for block_id in blocks.keys():
            t_block_start = time.perf_counter()

            exp_logger.block_start(block_id)

            raw_data = load_raw_data(block_id)
            filtered = filter_data(raw_data, blocks, block_id)
            ips_seqs = get_corpus(filtered, **params['corpus'])
            corpus_stats = corpus_basic_stats(ips_seqs)

            model = TorchWord2Vec(**params['word2vec'])

            t_train_start = time.perf_counter()
            model.train(
                ips_seqs,
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
            vector_size = getattr(model.model.wv, "vector_size", None) or getattr(
                model.model, "vector_size", 0
            )

            t_block_end = time.perf_counter()
            print(
                f"{info_prefix}"
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

            print("wv:", model.model.wv)

        # incremental の run 全体に対して 1 回だけ run_id を記録する
        if mapping_path is not None and alpha is not None and attack is not None:
            append_alpha_mapping(mapping_path, alpha, mode, attack, exp_logger.run_id, radius, normal_pull_lambda)

    else:
        raise ValueError(f"Unknown mode: {mode}")

# =====================================================
# メイン：攻撃 / α / mode をスイープ
# =====================================================
if __name__ == "__main__":
    # 「まったくスイープしない」場合:
    # DO_ATTACK_SWEEP = False
    # DO_ALPHA_SWEEP  = False
    # DO_MODE_SWEEP   = False
    # としておけば、ほぼ元の挙動に近い形で 1run だけ実行される

    # 攻撃ラベルのリスト
    if DO_ATTACK_SWEEP:
        attacks = ATTACK_LIST
    else:
        # config.py に書かれている ATTACK だけを使う
        attacks = [ATTACK]

    # モード
    modes = RUN_MODES if DO_MODE_SWEEP else [TRAINING_MODE]

    # α の候補
    if DO_ALPHA_SWEEP:
        alpha_list = ALPHAS
    else:
        alpha_list = [BASE_PARAMS['word2vec']['alpha_anom']]

    # 1組み合わせあたりの繰り返し回数
    REPEAT = 5

    for attack in attacks:
        # この ATTACK 用の BLOCKS を構築
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
                    print(f"\n===== attack={attack}, mode={mode}, alpha={alpha} =====")

                    params = deepcopy(BASE_PARAMS)
                    params['word2vec']['alpha_anom'] = alpha

                    for R in R_list:
                        print(f"\n----- norm_radius={R} -----")
                        if R is not None:
                            params['word2vec']['norm_radius'] = R
                        else:
                            if 'norm_radius' in params['word2vec']:
                                del params['word2vec']['norm_radius']

                        for normal_pull_lambda in normal_pull_lambda_list:
                            print(f"\n----- normal_pull_lambda={normal_pull_lambda} -----")
                            params['word2vec']['normal_pull_lambda'] = normal_pull_lambda

                            run_training(
                                mode=mode,
                                params=params,
                                blocks=blocks,
                                attack=attack,
                                alpha=alpha,
                                mapping_path=ALPHA_MAPPING_PATH if DO_ALPHA_SWEEP else None,
                                radius=R,
                                normal_pull_lambda=normal_pull_lambda,
                            )
