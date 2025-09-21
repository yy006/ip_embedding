import json
from pathlib import Path

###############################################################################
# Global path of the raw darknet traces
###############################################################################
breakpoint()
ROOT = Path(__file__).resolve().parents[1]
print("ROOT:", ROOT)
DATA = ROOT/'datasets'
TRACES  = f'{DATA}/top5000.csv'
#MODELS = f'{DATA}/models'
#GRAPHS = f'{DATA}/graphs'
#DATASETS = f'{DATA}/interim'
SERVICES = f'{DATA}/services/services.json'
#GT = f'{DATA}/groundtruth/ground_truth_full.csv.gz'
#MANUAL_GT = f'{DATA}/groundtruth/manual_gt.csv'

###############################################################################
# Domain knowledge based services
###############################################################################
with open(SERVICES, 'r') as file:
    LANGUAGES = json.loads(file.read())
    

LOWER_BOUND = '20210302'
