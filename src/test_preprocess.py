# tests/test_preprocess_main.py
import sys
import types
import importlib
from pathlib import Path

import pandas as pd
import pytest


# ---- テスト用ダミーPool（逐次実行にする） ----------------------------
class DummyPool:
    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return list(map(func, iterable))

    def close(self):
        pass

    def join(self):
        pass


# ---- 共通フィクスチャ：configをモックしてから preprocess を読み直す ----
@pytest.fixture
def prep(tmp_path, monkeypatch):
    """
    - config(TRACES, LOWER_BOUND) をテスト用にモック
    - multiprocessing.Pool を DummyPool に差し替え
    - preprocess モジュールをリロードして返す
    """
    # 1) ダミーconfigを作成
    cfg = types.ModuleType("config")
    cfg.TRACES = str(tmp_path)        # すべての入出力は tmp_path 配下
    cfg.LOWER_BOUND = "19700101"      # ループ打ち切りの下限（十分古い日付）
    sys.modules["config"] = cfg

    # 2) 既に import 済みなら消して読み直す
    if "preprocess" in sys.modules:
        del sys.modules["preprocess"]

    # 3) インポート
    preprocess = importlib.import_module("preprocess")

    # 4) Pool をダミーに差し替え（map が逐次化される）
    monkeypatch.setattr(preprocess, "Pool", DummyPool, raising=True)

    return preprocess


# ---- 補助：スペース区切りCSVを書くヘルパ ----------------------------
def write_raw_csv(path: Path, rows):
    """
    rows: list of dicts with keys ['ts','src_ip','dst_port','proto','pck_len']
    """
    df = pd.DataFrame(rows, columns=["ts", "src_ip", "dst_port", "proto", "pck_len"])
    df.to_csv(path, sep=" ", index=False)


def write_srcip_csv(path: Path, src_ips):
    """
    src_ips: list of 'src_ip' strings（列は 'src_ip' 1つだけのCSVを作る）
    """
    df = pd.DataFrame({"src_ip": src_ips})
    df.to_csv(path, sep=" ", index=False)


# ---- テスト：load_raw_data の基本動作 -------------------------------
def test_load_raw_data_parses_and_maps_protocols(prep, tmp_path):
    day = "20250101"

    # TRACES/day* に原始CSV（スペース区切り）を2つ作る
    f1 = tmp_path / f"{day}_part1.csv"
    f2 = tmp_path / f"{day}_part2.csv"

    write_raw_csv(
        f1,
        [
            # 2025-01-01 00:00:00
            {"ts": 1735689600, "src_ip": "10.0.0.1", "dst_port": 80, "proto": 6, "pck_len": 100},
            {"ts": 1735689660, "src_ip": "9.9.9.9", "dst_port": 53, "proto": 17, "pck_len": 60},
        ],
    )
    write_raw_csv(
        f2,
        [
            {"ts": 1735689720, "src_ip": "10.0.0.1", "dst_port": 443, "proto": 6, "pck_len": 120},
            {"ts": 1735689780, "src_ip": "8.8.8.8", "dst_port": 0, "proto": 1, "pck_len": 48},
        ],
    )

    raw = prep.load_raw_data(day)

    # レコード数（2+2）
    assert len(raw) == 4

    # 列名が想定どおり（get_dataの変換結果）
    for col in ["ts", "ip", "port", "proto", "pck_len", "pp"]:
        assert col in raw.columns

    # プロトコルの数値→文字列マッピング
    assert set(raw["proto"].unique()) <= {"tcp", "udp", "icmp", "oth"}

    # pp = "port/proto"
    sample = raw.iloc[0]
    assert sample["pp"] == f"{sample['port']}/{sample['proto']}"

    # ts は datetime 化されている
    assert pd.api.types.is_datetime64_any_dtype(raw["ts"])


# ---- テスト：filter_data が頻出IPだけに絞る & DatetimeIndex ------------
def test_filter_data_keeps_only_frequent_ips_and_sorts(prep, tmp_path):
    """
    load_filter_from_chunk(day) は get_files_from(day) で過去30日分を集計し、
    'src_ip' の出現回数>=10 を残す仕様。
    ここでは day 本体の raw ファイルに加えて、前日ファイルに '10.0.0.1' を大量投入して
    10回以上に到達させる。
    """
    day = "20250101"
    prev = "20241231"  # 過去1日分

    # --- day 当日の raw ファイル（2件中 '10.0.0.1' を1回, '9.9.9.9' も1回含める）
    f_day = tmp_path / f"{day}_part.csv"
    write_raw_csv(
        f_day,
        [
            {"ts": 1735689600, "src_ip": "10.0.0.1", "dst_port": 80, "proto": 6, "pck_len": 100},
            {"ts": 1735689660, "src_ip": "9.9.9.9", "dst_port": 53, "proto": 17, "pck_len": 60},
            {"ts": 1735689720, "src_ip": "10.0.0.1", "dst_port": 443, "proto": 6, "pck_len": 120},
        ],
    )

    # --- 前日ファイル：src_ip だけのCSVを作り、'10.0.0.1' を9回含める（合計で >=10 に到達）
    f_prev = tmp_path / f"{prev}_srcips.csv"
    write_srcip_csv(f_prev, ["10.0.0.1"] * 9 + ["9.9.9.9"] * 2)

    # raw をロード（day のみ対象）
    raw = prep.load_raw_data(day)

    # filter 実行（day を指定）
    filtered = prep.filter_data(raw, day)

    # 10回以上出現した '10.0.0.1' のみが残るはず
    assert set(filtered["ip"].unique()) == {"10.0.0.1"}

    # インデックスは DatetimeIndex で昇順になっている
    assert isinstance(filtered.index, pd.DatetimeIndex)
    assert filtered.index.is_monotonic_increasing

    # 元の raw の中で '10.0.0.1' の行数と一致
    expected_rows = (raw["ip"] == "10.0.0.1").sum()
    assert len(filtered) == expected_rows


# ---- テスト：get_next_day / get_prev_day --------------------------------
def test_get_next_and_prev_day(prep):
    assert prep.get_next_day("20201231") == "20210101"
    assert prep.get_prev_day("20250101") == "20241231"
