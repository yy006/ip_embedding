from multiprocessing import Pool, cpu_count
import pandas as pd
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
        "sep": ','
    }
}

def get_schema(dataset: str) -> Dict[str, Any]:
    """データセット名から read_csv 用のスキーマを返す。存在しなければわかりやすく失敗。"""
    try:
        return SCHEMA_REGISTRY[dataset]
    except KeyError as e:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(SCHEMA_REGISTRY)}") from e

def pool_setup(blocks: dict[int, Path]):
    items = sorted(blocks.items()) 
    cpus = len(blocks)
    if len(blocks)>cpu_count(): cpus = cpu_count()
    try:
        pool = Pool(processes=cpus)
    except ValueError:
        pool = Pool(processes=1)
    
    return pool, iter(items)

def get_data(item: Tuple[int, Path]) -> pd.DataFrame:
    block_id, path = item
    schema = get_schema(DATASET)

    # Read a single file
    print(pd.read_csv(path, nrows=5).columns)
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
    
    return f_df.assign(block=block_id)

###############################################################################
# Filtering preliminary preprocessed data
###############################################################################
def get_files_from(_date):
    """Load a list of file from the starting day to the previous 30th one.

    Parameters
    ----------
    _date : str
        starting date of file loading

    Returns
    -------
    list
        list of files to load

    """
    start = datetime.strptime(_date, '%Y%m%d')
    flist = []
    
    for d in range(30):
        target = start-timedelta(days=d)
        target = target.strftime('%Y%m%d')
        
        for fs in glob(f'{TRACES}/{target}*'):
            flist.append(fs)
        if target == LOWER_BOUND: break
    
    return flist

def count_daily_ips(x):
    df = pd.read_csv(x, sep=' ')['src_ip']
    
    return df

def load_filter_from_chunk(day):
    pool, iterable = pool_setup(get_files_from(day))
    df_list = pool.map(count_daily_ips, iterable)
    pool.close()
    counts = pd.concat(df_list).reset_index().value_counts('src_ip')

    return set(counts[counts>=10].index)


###############################################################################
# Main functions
###############################################################################
def load_raw_data(blocks: dict[int, Path]):
    pool, iterable = pool_setup(blocks)
    df_list = pool.map(get_data, iterable)
    pool.close()
    pool.join()
    raw_data = pd.concat(df_list)
    return raw_data
breakpoint()

def filter_data(raw_data, day_to_filter):
    #10回以上出現するIPアドレスを抽出
    filt = load_filter_from_chunk(day_to_filter)
    # Filter IPS
    filtered = raw_data[raw_data.ip.isin(set(filt))]
    # Datetime index
    filtered.index = pd.DatetimeIndex(filtered.ts)
    filtered = filtered.sort_index()
        
    return filtered

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