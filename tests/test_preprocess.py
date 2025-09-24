import json
import importlib
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


class TestMainFunction:
    def test_main(self):
        assert 10==10