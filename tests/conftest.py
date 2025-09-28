import sys, types
from pathlib import Path

def pytest_sessionstart(session):
    # テスト用の軽い config モジュールを作って sys.modules に登録
    fake = types.ModuleType("config")
    fake.ROOT = Path(__file__).resolve().parents[1]
    fake.DATASET = "UNSW-NB15"
    fake.DATA = fake.ROOT/'datasets'/fake.DATASET
    fake.BLOCKS = {1: fake.DATA/"top30.csv", 2: fake.DATA/"top30.csv_2"}
    fake.SERVICES = fake.DATA/'services/services.json'
    fake.TRAINING_MODE = "incremental"
    sys.modules["config"] = fake