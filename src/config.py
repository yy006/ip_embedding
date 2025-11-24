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
'''
ATTACK = "Exploits"  # "Worms" | "DoS" | "Analysis" | "Backdoor" | "Exploits" | "Fuzzers" | "Generic" | "Reconnaissance" | "Shellcode" | "Worms"
PREPROCESS = "UNSW-NB15_2_by2h/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h"

DATA = ROOT/'datasets'/DATASET
DATA_PATH = DATA / PREPROCESS / ATTACK

# ブロック番号 → ファイルパス（Path）
BLOCKS: dict[int, Path] = {
    1: DATA_PATH / f"2015012218_2015012220_by2h_{ATTACK}.csv",
    2: DATA_PATH / f"2015012220_2015012222_by2h_{ATTACK}.csv",
    3: DATA_PATH / f"2015012222_2015012300_by2h_{ATTACK}.csv",
    4: DATA_PATH / f"2015012300_2015012302_by2h_{ATTACK}.csv",
    5: DATA_PATH / f"2015021800_2015021802_by2h_{ATTACK}.csv",
    6: DATA_PATH / f"2015021802_2015021804_by2h_{ATTACK}.csv",
}
'''
DATA = ROOT/'datasets'/DATASET
DATA_PATH = DATA / "UNSW-NB15_2_by2h/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h"
BLOCKS: dict[int, Path] = {
    1: DATA_PATH / "2015012218_2015012220_by2h.csv",
    2: DATA_PATH / "2015012220_2015012222_by2h.csv",
    3: DATA_PATH / "2015012222_2015012300_by2h.csv",
    4: DATA_PATH / "2015012300_2015012302_by2h.csv",
    5: DATA_PATH / "Backdoor" /  "2015021800_2015021802_by2h_Backdoor.csv",
    #6: DATA_PATH / "DoS" / "2015021802_2015021804_by2h_DoS.csv",
}

#MODELS = f'{DATA}/models'
#GRAPHS = f'{DATA}/graphs'
#DATASETS = f'{DATA}/interim'
SERVICES = f'{DATA}/services/services.json'

TrainingMode = Literal["single", "incremental"]
TRAINING_MODE: TrainingMode = 'single'  # "single" or "incremental"
#GT = f'{DATA}/groundtruth/ground_truth_full.csv.gz'
#MANUAL_GT = f'{DATA}/groundtruth/manual_gt.csv'

ARTIFACTS_ROOT = Path(ROOT) / "experiments"


###############################################################################
# Domain knowledge based services
###############################################################################
with open(SERVICES, 'r') as file:
    LANGUAGES = json.loads(file.read())

