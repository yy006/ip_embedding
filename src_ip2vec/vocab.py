from collections import Counter
import numpy as np
import pandas as pd

def build_vocab_from_df_ip2vec(
    df: pd.DataFrame,
    *,
    src_col="srcip",
    dst_col="dstip",
    dport_col="dsport",
    proto_col="proto",
    min_count=0,
    use_prefix=True,
):
    def tok(kind, v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):  # NaN
            return None
        if kind == "DPORT":
            try:
                v = int(v)
            except Exception:
                pass
        return f"{kind}:{v}" if use_prefix else str(v)

    counter = Counter()
    for _, r in df.iterrows():
        for t in [
            tok("SRCIP", r.get(src_col)),
            tok("DSTIP", r.get(dst_col)),
            tok("DPORT", r.get(dport_col)),
            tok("PROTO", r.get(proto_col)),
        ]:
            if t is not None:
                counter[t] += 1

    tokens = [t for t, f in counter.items() if f >= min_count]
    tokens.sort()

    token2id = {t: i for i, t in enumerate(tokens)}
    id2token = {i: t for t, i in token2id.items()}

    freqs = np.array([counter[t] for t in tokens], dtype=np.float64)
    freqs = freqs ** 0.75
    freqs = freqs / freqs.sum()

    return token2id, id2token, freqs
