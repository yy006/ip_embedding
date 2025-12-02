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

'''
DATA = ROOT/'datasets'/DATASET
DATA_PATH = DATA / "UNSW-NB15_2_by0.5h"
BLOCKS: dict[int, Path] = {
    1: DATA_PATH / "2015012219_2015012220_by0.5h.csv",
    2: DATA_PATH / "2015012220_2015012220_by0.5h.csv",
    3: DATA_PATH / "2015012220_2015012221_by0.5h.csv",
    4: DATA_PATH / "2015012221_2015012221_by0.5h.csv",
    5: DATA_PATH / "2015012221_2015012222_by0.5h.csv",
    6: DATA_PATH / "2015012222_2015012222_by0.5h.csv",
    7: DATA_PATH / "2015012222_2015012223_by0.5h.csv",
    8: DATA_PATH / "2015012223_2015012223_by0.5h.csv",
    9: DATA_PATH / "2015012223_2015012300_by0.5h.csv",
    10: DATA_PATH / "2015012300_2015012300_by0.5h.csv",
    11: DATA_PATH / "2015021800_2015021800_by0.5h.csv",
    #12: DATA_PATH / "2015021800_2015021801_by0.5h.csv",
    #13: DATA_PATH / "2015021801_2015021801_by0.5h.csv",
    #14: DATA_PATH / "2015021801_2015021802_by0.5h.csv",
    #15: DATA_PATH / "2015021802_2015021802_by0.5h.csv",
    #16: DATA_PATH / "2015021802_2015021803_by0.5h.csv",
    #17: DATA_PATH / "2015021803_2015021803_by0.5h.csv",
    #18: DATA_PATH / "2015021803_2015021804_by0.5h.csv",
}
'''
'''
DATA = ROOT/'datasets'/DATASET
DATA_PATH = DATA / "UNSW-NB15_2_by2h/UNSW-NB15_2_ipmap59to175_drop175benign_with_class_name_by2h"
BLOCKS: dict[int, Path] = {
    1: DATA_PATH / "2015012218_2015012220_by2h.csv",
    2: DATA_PATH / "2015012220_2015012222_by2h.csv",
    3: DATA_PATH / "2015012222_2015012300_by2h.csv",
    4: DATA_PATH / "2015012300_2015012302_by2h.csv",
    5: DATA_PATH / "2015021800_2015021802_by2h.csv",
    6: DATA_PATH / "2015021802_2015021804_by2h.csv",
}
'''
'''
DATA = ROOT/'datasets'/DATASET
DATA_PATH = DATA / "UNSW-NB15_2_by1h"
BLOCKS: dict[int, Path] = {
    1: DATA_PATH / "2015012218_2015012220_by2h.csv",
    2: DATA_PATH / "2015012220_2015012221_by1h.csv",
    3: DATA_PATH / "2015012221_2015012222_by1h.csv",
    4: DATA_PATH / "2015012222_2015012223_by1h.csv",
    5: DATA_PATH / "2015012223_2015012300_by1h.csv",
    6: DATA_PATH / "2015012300_2015012302_by2h.csv",
    #7: DATA_PATH / "2015021800_2015021801_by1h.csv",
    #8: DATA_PATH / "2015021801_2015021802_by1h.csv",
    #9: DATA_PATH / "2015021802_2015021803_by1h.csv",
    #10: DATA_PATH / "2015021803_2015021804_by1h.csv",
}
'''

DATA = ROOT/'datasets'/DATASET
DATA_PATH = DATA / "UNSW-NB15_2_by10h"
BLOCKS: dict[int, Path] = {
    1: DATA_PATH / "2015012218_2015012220_by2h_product_5.csv",
    2: DATA_PATH / "2015012220_2015012222_by2h_product_5.csv",
    3: DATA_PATH / "2015012222_2015012300_by2h_product_5.csv",
    4: DATA_PATH / "2015012300_2015012302_by2h_product_5.csv",
    5: DATA_PATH / "2015021800_2015021802_by2h_product_5.csv",
    6: DATA_PATH / "2015021802_2015021804_by2h_product_5.csv",
}


SERVICES = f'{DATA}/services/services.json'

TrainingMode = Literal["single", "incremental"]
TRAINING_MODE: TrainingMode = 'incremental'  # "single" or "incremental"

ARTIFACTS_ROOT = Path(ROOT) / "experiments"


###############################################################################
# Domain knowledge based services
###############################################################################
with open(SERVICES, 'r') as file:
    LANGUAGES = json.loads(file.read())

