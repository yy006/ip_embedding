import json
import importlib
from pathlib import Path
import pytest

def build_block_file_map():
    return {
        1: Path("datasets/top5000.csv"),
        2: Path("datasets/top5000.csv"),
        3: Path("datasets/top5000.csv"),
    }

class TestLoadData:
    def test_pool_setup(self):
        


class TestMainFunction:
    def test_main(self):
        assert 10==10