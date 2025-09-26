import json
from pathlib import Path
from typing import Literal

###############################################################################
# Global path of the raw darknet traces
###############################################################################
#breakpoint()
ROOT = Path(__file__).resolve().parents[1]
print("ROOT:", ROOT)
DATASET = 'UNSW-NB15'
DATA = ROOT/'datasets'/DATASET

TRACES = f'{DATA}/top5000.csv'
# ブロック番号 → ファイルパス（Path）
BLOCKS: dict[int, Path] = {
    1: DATA / "top5000.csv",
    2: DATA / "top5000.csv",
    3: DATA / "top5000.csv",
}
#MODELS = f'{DATA}/models'
#GRAPHS = f'{DATA}/graphs'
#DATASETS = f'{DATA}/interim'
SERVICES = f'{DATA}/services/services.json'

TrainingMode = Literal["single", "incremental"]
TRAINING_MODE: TrainingMode = "incremental"
#GT = f'{DATA}/groundtruth/ground_truth_full.csv.gz'
#MANUAL_GT = f'{DATA}/groundtruth/manual_gt.csv'


###############################################################################
# Domain knowledge based services
###############################################################################
with open(SERVICES, 'r') as file:
    LANGUAGES = json.loads(file.read())

