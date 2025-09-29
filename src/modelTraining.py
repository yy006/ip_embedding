from config import *
from preprocess import load_raw_data, filter_data, get_next_day
from corpus import get_corpus
from word2vec import Word2Vec
import numpy as np
import time
from logger import ExperimentLogger, save_model_and_dict, corpus_basic_stats


SAVE = True

params = [{'corpus':{'services':'auto', 'without_duplicates':True, 'top_ports':300},
       'word2vec':{'c':25, 'e':50, 'epochs':3, 'method':'incremental'}}]

print(TRAINING_MODE)
print(BLOCKS)

# === 実験ロガー初期化 ===
artifact_root = ARTIFACTS_ROOT
exp_logger = ExperimentLogger(
    artifact_root=artifact_root,
    dataset=DATASET,
    mode=TRAINING_MODE,
    blocks=BLOCKS,
    params=params[0],
)

if TRAINING_MODE == "single":
    exp_logger.block_start(1)

    raw_data = load_raw_data(len(BLOCKS))
    filtered = filter_data(raw_data, BLOCKS, len(BLOCKS))
    corpus = get_corpus(filtered, **params[0]['corpus'])
    corpus_stats = corpus_basic_stats(corpus)

    model = Word2Vec(**params[0]['word2vec'])
    model.train(corpus, save=SAVE)

    # === 実験ログ記録 ===
    # モデル・辞書の保存先パス
    mpath = exp_logger.model_path(1)
    dpath = exp_logger.dict_path(1)

    # モデル・辞書の保存
    save_model_and_dict(model, {}, mpath, dpath)

    vocab_size = len(model.model.wv.index_to_key)
    vector_size = getattr(model.model.wv, "vector_size", None) or getattr(model.model, "vector_size", 0)

    exp_logger.block_end(
        1,
        vocab_size=vocab_size,
        vector_size=vector_size,
        corpus_stats=corpus_stats,
        )
    exp_logger.finalize()

    print("wv:", model.model.wv)

elif TRAINING_MODE == "incremental":
    for block_id in BLOCKS.keys():
        exp_logger.block_start(block_id)

        raw_data = load_raw_data(block_id)
        filtered = filter_data(raw_data, BLOCKS, block_id)
        corpus = get_corpus(filtered, **params[0]['corpus'])
        corpus_stats = corpus_basic_stats(corpus)

        model = Word2Vec(**params[0]['word2vec'])
        model.train(corpus, save=SAVE)

        # === 実験ログ記録 ===
        # モデル・辞書の保存先パス
        mpath = exp_logger.model_path(block_id)
        dpath = exp_logger.dict_path(block_id)
    
        # モデル・辞書の保存
        save_model_and_dict(model, {}, mpath, dpath)

        vocab_size = len(model.model.wv.index_to_key)
        vector_size = getattr(model.model.wv, "vector_size", None) or getattr(model.model, "vector_size", 0)

        exp_logger.block_end(
            block_id,
            vocab_size=vocab_size,
            vector_size=vector_size,
            corpus_stats=corpus_stats,
            )
        exp_logger.finalize()

        print("wv:", model.model.wv)
        #print("key:", model.model.wv.index_to_key)
