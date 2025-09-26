from multiprocessing import Pool, cpu_count
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from glob import glob
from config import *
from typing import Tuple, Dict, Any

###############################################################################
# Raw data loading and preliminary transformations
###############################################################################

SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "UNSW-NB15": {
        "usecols": ['Timestamp', 'Source IP', 'Destination Port', 'Protocol', 'Total Length of Fwd Packets'],
        "rename": {'Timestamp': 'ts', 'Source IP': 'ip', 'Destination Port': 'port', 'Protocol': 'proto'},
        "sep": ',',
        "ip_col": ['Source IP'],
    }
}

def get_schema(dataset: str) -> Dict[str, Any]:
    """データセット名から read_csv 用のスキーマを返す。存在しなければわかりやすく失敗。"""
    try:
        return SCHEMA_REGISTRY[dataset]
    except KeyError as e:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(SCHEMA_REGISTRY)}") from e

def pool_setup(flist: list) -> Tuple[Pool, Any]:
    cpus = len(flist)
    if len(flist)>cpu_count(): cpus = cpu_count()
    try:
        pool = Pool(processes=cpus)
    except ValueError:
        pool = Pool(processes=1)
    
    return pool, iter(flist)

def get_data(path: Path) -> pd.DataFrame:
    schema = get_schema(DATASET)

    # Read a single file
    #print(pd.read_csv(path, nrows=5).columns)
    f_df = pd.read_csv(path,
                       sep=schema["sep"],
                       usecols=schema["usecols"],
                       skipinitialspace=True, 
                       ).rename(columns=schema['rename'])

    # Replace decimal representation of protocols to string identifier
    to_replace = dict()
    for x in f_df.proto.unique():
        if x == 6: to_replace[x] = 'tcp'
        elif x == 17: to_replace[x] = 'udp'
        elif x == 1: to_replace[x] = 'icmp'
        else: to_replace[x] = 'oth'
    f_df.proto = f_df.proto.replace(to_replace)
    # Merge port and protocol as 'port/protocol'
    f_df['pp'] = f_df.port.astype(str)+"/"+f_df.proto
    # Convert timestamps
    #f_df.ts = f_df.ts.apply(lambda x: datetime.fromtimestamp(x))
    
    return f_df

###############################################################################
# Filtering preliminary preprocessed data
###############################################################################
def get_files_from(_blocks: dict[int, Path]):
    """Load a list of file from the starting block to the previous.

    Parameters
    ----------
    _date : str
        starting block of file loading

    Returns
    -------
    list
        list of files to load

    """

    flist = []
    
    for i in range(len(_blocks)):
        
        flist.append(_blocks[i+1])
        
        #for fs in glob(f'{TRACES}/{target}*'):
    return flist

def count_block_ips(x):
    # read_csvの引数をschemaから取る
    schema = get_schema(DATASET)

    df = pd.read_csv(x,
                     sep=schema['sep'],
                     usecols=schema['ip_col'],
                     skipinitialspace=True)
    
    return df

def load_filter_from_chunk(blocks):
    #flist = sorted(blocks.items()) 
    schema = get_schema(DATASET)

    pool, iterable = pool_setup(get_files_from(blocks))
    df_list = pool.map(count_block_ips, iterable)
    pool.close()
    counts = pd.concat(df_list).reset_index().value_counts(schema['ip_col'][0])

    return set(counts[counts>=10].index)


###############################################################################
# Main functions
###############################################################################
def load_raw_data(block_number):
    if TRAINING_MODE == "single":
        flist = [BLOCKS[k] for k in block_number]

    elif TRAINING_MODE == "incremental":
        flist = [BLOCKS[block_number]]

    pool, iterable = pool_setup(flist)
    df_list = pool.map(get_data, iterable)
    pool.close()
    pool.join()
    raw_data = pd.concat(df_list)
    return raw_data

breakpoint()

def filter_data(raw_data, block_to_filter):
    #10回以上出現するIPアドレスを抽出
    filt = load_filter_from_chunk(block_to_filter)
    # Filter IPS
    filtered = raw_data[raw_data.ip.isin(set(filt))]
    # Datetime index (TODO: datetime index処理が必要かの考慮)
    filtered.index = pd.DatetimeIndex(filtered.ts)
    filtered = filtered.sort_index()
        
    return filtered

#breakpoint()

def get_next_day(start):
    start = datetime.strptime(start, '%Y%m%d')
    day = start+timedelta(days=1)
    day = day.strftime('%Y%m%d')
    
    return day

def get_prev_day(start):
    start = datetime.strptime(start, '%Y%m%d')
    day = start-timedelta(days=1)
    day = day.strftime('%Y%m%d')
    
    return day