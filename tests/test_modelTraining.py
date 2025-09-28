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
                'vector_size': 5,
                'window': 5,
                'min_count': 1,
                'workers': 1,
                'epochs': 10,
                'mname': 'test_model'
            }
        }
        SAVE = False  # テスト中はモデルを保存しない

        raw_data = prep.load_raw_data(1)
        filtered_df = prep.filter_data(raw_data, BLOCKS, 1)

        corpus = corpus.get_corpus(filtered_df, services='auto', without_duplicates=False, top_ports=3)

        model = Word2Vec(**params['word2vec'])
        model.train(corpus, save=SAVE)
        print(model.model)
