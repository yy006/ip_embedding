import numpy as np
from pandas import DatetimeIndex
from config import LANGUAGES 

def drop_duplicates(ips_seqs, label_seqs=None):
    """
    連続する同一 IP を削除する。
    - label_seqs が与えられている場合は、IP を削った位置に対応する Label も一緒に削る。
    - label_seqs が None の場合は、従来どおり IP のみ処理して返す。

    Parameters
    ----------
    ips_seqs : list[list[str]]
        各サービス/時間窓ごとの IP 列
    label_seqs : list[list[int]] | None
        各 IP に対応するラベル列 (任意)

    Returns
    -------
    if label_seqs is None:
        new_ips_seqs
    else:
        new_ips_seqs, new_label_seqs
    """
    new_ips_seqs = []
    new_label_seqs = [] if label_seqs is not None else None

    # ラベル付きの場合は zip でペアにして処理、ラベルなしなら None をダミーにして回す
    if label_seqs is None:
        iter_pairs = zip(ips_seqs, [None] * len(ips_seqs))
    else:
        iter_pairs = zip(ips_seqs, label_seqs)

    for ips, labs in iter_pairs:
        ips_arr = np.array(ips)

        # 次の要素に1つシフトして末尾だけダミーを入れる（元実装と同じアイデア）
        nxt_ips = np.roll(ips_arr, -1)
        nxt_ips[-1] = "NULL"

        # 連続する同一IPを落とすマスク
        mask = ips_arr != nxt_ips

        new_ips_seqs.append(list(ips_arr[mask]))

        if labs is not None:
            labs_arr = np.array(labs)
            new_label_seqs.append(list(labs_arr[mask]))

    if label_seqs is None:
        return new_ips_seqs
    else:
        return new_ips_seqs, new_label_seqs

def get_top_ports(dev, TOP):
    try: dev = dev.drop(columns=['serv'])
    except: pass
    topports = dev.value_counts('pp')
    top_p = topports.iloc[:TOP].index
    temp__ = dev.drop(columns=['ts']).reset_index()
    idx = temp__[temp__.pp.isin(top_p)].index
    temp__.loc[idx,'serv'] = temp__.loc[idx,'pp']
    temp__ = temp__.fillna('other')
    temp__.index = DatetimeIndex(temp__.ts)
    
    return temp__

def get_services(x):
    if x in LANGUAGES: 
        return LANGUAGES[x]
    else: 
        x = x.split('/')[0]
        if x!='-':
            if int(x) >= 0 and int(x) <= 1023: 
                return 'unk_sys'
            elif int(x) >= 1024 and int(x) <= 49151: 
                return 'unk_usr'
            elif int(x) >= 49152 and int(x) <= 65535: 
                return 'unk_eph'
        else: 
            return 'icmp'

def get_hours(x):
    hh = x.hour
    if hh < 10: hh = f'0{hh}'
    dd = x.day
    if dd < 10: dd = f'0{dd}'
    mm = x.month
    if mm < 10: mm = f'0{mm}'
    yy = x.year
    if yy < 10: yy = f'0{yy}'
    
    return f'{yy}_{mm}_{dd}_{hh}'


def get_corpus(data, without_duplicates=True, services='auto', top_ports=None):
    # Define 1h sequences
    data['hour'] = data.ts.apply(get_hours)
    
    if services=='single':
        rows = data.groupby(['hour']).agg({'ip':list})\
                   .sort_values(['hour']).values
        corpus = [x[0] for x in rows]

    elif services=='auto':
        if not isinstance(top_ports, int):
            raise Exception('top_ports parameter missing. Provide the number '\
                            'of top ports to use as services')
        data = get_top_ports(data, top_ports)
        rows = data.groupby(['serv', 'hour']).agg({'ip':list, 'Label':list})\
                   .sort_values(['hour', 'serv'])
        corpus = [x[0] for x in rows]

        ips_seqs   = rows['ip'].tolist()

    elif services=='hybrid':
        if not isinstance(top_ports, int):
            raise Exception('top_ports parameter missing. Provide the number '\
                            'of top ports to use as services')
        data = get_top_ports(data, top_ports)
        rows1 = data.groupby(['serv', 'hour']).agg({'ip':list})\
                .sort_values(['hour', 'serv']).values
        corpus1 = [x[0] for x in rows1]
        
        data['serv'] = data.pp.apply(get_services)
        rows2 = data.groupby(['serv', 'hour']).agg({'ip':list})\
                .sort_values(['hour', 'serv']).values
        corpus2 = [x[0] for x in rows2]
        corpus = corpus1 + corpus2

    elif services=='dks':
        data['serv'] = data.pp.apply(get_services)
        rows = data.groupby(['serv', 'hour']).agg({'ip':list})\
                   .sort_values(['hour', 'serv']).values
        corpus = [x[0] for x in rows]

    if without_duplicates:
        ips_seqs = drop_duplicates(ips_seqs)

    return ips_seqs