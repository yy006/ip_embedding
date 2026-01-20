from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

class IP2VecPairDataset(Dataset):
    """
    1行(1通信レコード)から、IP2Vecの固定ルールで (center, context) ペアを生成。

    ルール:
      srcip -> dstip, dsport, proto
      dstip -> dsport
      dstip -> proto

    __getitem__ はその行から作れるペア全部を返す（可変長）。
    collate_fn でフラット化して学習に渡す。
    """
    def __init__(
        self,
        df: pd.DataFrame,
        token2id: dict,
        *,
        src_col="srcip",
        dst_col="dstip",
        dport_col="dsport",
        proto_col="proto",
        use_prefix=True,
    ):
        self.df = df.reset_index(drop=True)
        self.token2id = token2id
        self.src_col = src_col
        self.dst_col = dst_col
        self.dport_col = dport_col
        self.proto_col = proto_col
        self.use_prefix = use_prefix

    def __len__(self):
        return len(self.df)

    def _tok(self, kind: str, v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):  # NaN
            return None
        if kind == "DPORT":
            try:
                v = int(v)
            except Exception:
                pass
        return f"{kind}:{v}" if self.use_prefix else str(v)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        src = self._tok("SRCIP", r.get(self.src_col))
        dst = self._tok("DSTIP", r.get(self.dst_col))
        dp  = self._tok("DPORT", r.get(self.dport_col))
        pr  = self._tok("PROTO", r.get(self.proto_col))

        pairs = []

        # srcip -> dstip, dsport, proto
        if src and dst: pairs.append((src, dst))
        if src and dp:  pairs.append((src, dp))
        if src and pr:  pairs.append((src, pr))

        # dstip -> dsport
        if dst and dp:  pairs.append((dst, dp))

        # dstip -> proto
        if dst and pr:  pairs.append((dst, pr))

        # token -> id（語彙に無いものは落とす）
        c_ids, o_ids = [], []
        for c, o in pairs:
            ci = self.token2id.get(c)
            oi = self.token2id.get(o)
            if ci is None or oi is None:
                continue
            c_ids.append(ci)
            o_ids.append(oi)

        return torch.tensor(c_ids, dtype=torch.long), torch.tensor(o_ids, dtype=torch.long)


def ip2vec_collate_fn(batch):
    """
    batch: List[(center_ids_i, pos_ids_i)] where each is 1D tensor (Pi,)
    -> すべて結合して (B_pairs,), (B_pairs,) を返す
    """
    centers, pos = [], []
    for c, p in batch:
        if c.numel() == 0:
            continue
        centers.append(c)
        pos.append(p)

    if not centers:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    return torch.cat(centers, dim=0), torch.cat(pos, dim=0)
