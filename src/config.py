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

# ブロック番号 → ファイルパス（Path）
BLOCKS: dict[int, Path] = {
    1: DATA / "UNSW-NB15_2_5tuple_by2h_ipmap59to175_drop175benign"/"2015012218_2015012220_by2h.csv",
    2: DATA / "UNSW-NB15_2_5tuple_by2h_ipmap59to175_drop175benign"/"2015012220_2015012222_by2h.csv",
    3: DATA / "UNSW-NB15_2_5tuple_by2h_ipmap59to175_drop175benign"/"2015012222_2015012300_by2h.csv",
    4: DATA / "UNSW-NB15_2_5tuple_by2h_ipmap59to175_drop175benign"/"2015012300_2015012302_by2h.csv",
    5: DATA / "UNSW-NB15_2_5tuple_by2h_ipmap59to175_drop175benign"/"2015021800_2015021802_by2h.csv",
}
#MODELS = f'{DATA}/models'
#GRAPHS = f'{DATA}/graphs'
#DATASETS = f'{DATA}/interim'
SERVICES = f'{DATA}/services/services.json'

TrainingMode = Literal["single", "incremental"]
TRAINING_MODE: TrainingMode = "single"
#GT = f'{DATA}/groundtruth/ground_truth_full.csv.gz'
#MANUAL_GT = f'{DATA}/groundtruth/manual_gt.csv'

ARTIFACTS_ROOT = Path(ROOT) / "experiments"


###############################################################################
# Domain knowledge based services
###############################################################################
with open(SERVICES, 'r') as file:
    LANGUAGES = json.loads(file.read())

