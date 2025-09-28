import pytest
from pathlib import Path
import src.corpus as corpus
import src.preprocess as prep
from src.word2vec import Word2Vec


class TestModelTraining:
    def test_model_training_with_small_dataset(self):
        BLOCKS = {1: Path('datasets/UNSW-NB15/top30.csv'),
                  2: Path('datasets/UNSW-NB15/top30.csv_2')}
        params = {
            'word2vec': {
                'c': 4,
                'e': 5,
                'epochs': 3,
                'mname': 'test_model'
            }
        }
        SAVE = False  # テスト中はモデルを保存しない

        raw_data = prep.load_raw_data(1)
        filtered_df = prep.filter_data(raw_data, BLOCKS, 1)

        corpus_df = corpus.get_corpus(filtered_df, services='auto', without_duplicates=False, top_ports=3)

        model = Word2Vec(**params['word2vec'])
        model.train(corpus_df, save=SAVE)
        print(model.model.wv)
        print(model.model.wv.index_to_key)
        print(model.model.wv[['192.168.10.19']])
