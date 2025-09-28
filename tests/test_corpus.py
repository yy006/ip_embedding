import pytest
from pathlib import Path
import src.corpus as corpus
import src.preprocess as prep

class TestGetCorpus:
    # サービス定義がautoの場合
    def test_auto_services(self):
        BLOCKS = {1: Path('datasets/UNSW-NB15/top30.csv'),
                  2: Path('datasets/UNSW-NB15/top30.csv_2')}

        raw_data = prep.load_raw_data(1)
        filtered_df = prep.filter_data(raw_data, BLOCKS, 1)
        print(filtered_df)

        corpus_df = corpus.get_corpus(filtered_df, services='auto', without_duplicates=False, top_ports=3)

        # 上位3つのポートとその他の4サービスが定義されていて、それらが4文のコーパスとして出力されていることを確認した
        print(corpus_df)