from multiprocessing import Pool, cpu_count
import json
import importlib
import pandas as pd
from pathlib import Path
import pytest
import src.preprocess as m

class TestGetSchema:
    def test_returns_expected_schema_for_unsw_nb15(self):
        """
        スキーマに定義されたデータセット名を渡した時、期待される辞書が返ること
        """
        schema = m.get_schema("UNSW-NB15")
        assert schema == {
            "usecols": ['Timestamp', 'Source IP', 'Destination Port', 'Protocol', 'Total Length of Fwd Packets'],
            "rename": {'Timestamp': 'ts', 'Source IP': 'ip', 'Destination Port': 'port', 'Protocol': 'proto'},
            "sep": ',',
            "ip_col": ['Source IP'],
            }

    def test_raises_value_error_for_unknown_dataset(self):
        """
        スキーマに定義されていないデータセット名を渡した時、ValueErrorが発生すること
        """
        with pytest.raises(ValueError) as excinfo:
            m.get_schema("UNKNOWN-DATASET")
        assert "Unknown dataset 'UNKNOWN-DATASET'" in str(excinfo.value)

class TestPoolSetup:
    def test_pool_setup_with_few_files(self):
        flist = [Path(f"file_{i}.csv") for i in range(2)]

        pool, it = m.pool_setup(flist)
        print(pool)
        assert pool._processes == 2
        assert list(it) == flist
        pool.close()
        pool.join()

    def test_pool_setup_with_many_files(self):
        flist = [Path(f"file_{i}.csv") for i in range(100)]

        pool, it = m.pool_setup(flist)
        print(pool)
        assert pool._processes <= cpu_count()
        assert list(it) == flist
        pool.close()
        pool.join()
        
class TestGetData:
    def test_get_data_correctly_processes_file(self):
        """
        get_data が正しくデータを読み込み、前処理を行うこと
        """
        path = Path('datasets/UNSW-NB15/top30.csv')

        out_df = m.get_data(path)

        for col in ['ts', 'ip', 'port', 'proto', 'pp']:
            assert col in out_df.columns

        assert out_df['proto'].isin(['tcp', 'udp', 'icmp', 'oth']).all()

        # 未知のプロトコルが 'oth' にマッピングされていること(1行のみ追加してある)
        assert (out_df["proto"] == "oth").sum() == 1

        # 読み込んだデータが29行であること
        assert out_df.shape[0] == 29

class TestLoadRawData:
    def test_load_raw_data_correctly_loads_and_processes(self):
        """
        load_raw_data が正しくデータを読み込み、前処理を行うこと
        """

        out_df = m.load_raw_data(1)
        out_df_head = out_df.head(3)

        expected_head = pd.DataFrame({
            'ip': ['192.168.10.50', '192.168.10.50', '8.6.0.1'],
            'port': [389, 389, 0],
            'proto': ['tcp', 'tcp', 'oth'],
            'ts': ['6/7/2017 8:59', '6/7/2017 8:59', '6/7/2017 8:59'],
            'Total Length of Fwd Packets': [9668, 11364, 0],
            'pp': ['389/tcp', '389/tcp', '0/oth']
            })
        
        # 期待値の ts を datetime に変換（米式: 月/日/年）
        expected_head['ts'] = pd.to_datetime(expected_head['ts'], format='%m/%d/%Y %H:%M')
        
        assert out_df_head.equals(expected_head)
        assert out_df.shape == (29, 6)

class TestFilterData:
    def test_filter_data_correctly_filters_by_ip(self):
        """
        filter_data が正しくIPでフィルタリングを行うこと
        """
        BLOCKS = {1: Path('datasets/UNSW-NB15/top30.csv'),
                  2: Path('datasets/UNSW-NB15/top30.csv_2')}

        raw_df = m.load_raw_data(1)
        filtered_df = m.filter_data(raw_df, BLOCKS, 1)
        filtered_df_head = filtered_df.head(3)

        expected_head = pd.DataFrame({
            'ip': ['192.168.10.19', '192.168.10.19', '192.168.10.19'],
            'port': [5353, 53, 123],
            'proto': ['udp', 'udp', 'udp'],
            'ts': ['6/7/2017 9:00', '6/7/2017 9:00', '6/7/2017 9:00'],
            'Total Length of Fwd Packets': [26257, 46, 672],
            'pp': ['5353/udp', '53/udp', '123/udp']
            })
        expected_head = expected_head.set_index(pd.to_datetime(expected_head['ts']))
        # 期待値の ts を datetime に変換（米式: 月/日/年）
        expected_head['ts'] = pd.to_datetime(expected_head['ts'], format='%m/%d/%Y %H:%M')

        assert filtered_df_head.equals(expected_head)
        assert filtered_df.shape == (16, 6)
        assert filtered_df['ip'].unique() == '192.168.10.19'

