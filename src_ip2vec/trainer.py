from __future__ import annotations

from torch.utils.data import DataLoader
import pandas as pd
import torch

from .model import SkipGramNegSampling
from .dataset import IP2VecPairDataset, ip2vec_collate_fn
from .vocab import build_vocab_from_df_ip2vec


class TorchIP2Vec:
    def __init__(self, e=50, epochs=2, neg_k=5, lr=0.025, device=None, norm_radius=None):
        self.embedding_size = e
        self.epochs = epochs
        self.neg_k = neg_k
        self.lr = lr
        self.norm_radius = norm_radius

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.unigram_table: torch.Tensor | None = None

        self.model: SkipGramNegSampling | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    # -----------------------------
    # gensim 互換っぽい wv ラッパ
    # -----------------------------
    class _WVWrapper:
        def __init__(self, parent: "TorchIP2Vec"):
            self._p = parent

        @property
        def index_to_key(self):
            # id順
            return [self._p.id2token[i] for i in range(len(self._p.id2token))]

        @property
        def vector_size(self):
            return self._p.embedding_size

        def __getitem__(self, key):
            p = self._p
            if p.model is None:
                raise RuntimeError("model is None")

            if isinstance(key, str):
                idx = p.token2id.get(key)
                if idx is None:
                    raise KeyError(key)
            else:
                idx = int(key)

            with torch.no_grad():
                return p.model.in_embed.weight[idx].detach().cpu().numpy()

    def _attach_wv(self):
        """model.wv を付ける（評価コード互換用）"""
        if self.model is None:
            return
        self.model.wv = TorchIP2Vec._WVWrapper(self)

    # -----------------------------
    # 学習本体
    # -----------------------------
    def train_ip2vec(
        self,
        df: pd.DataFrame,
        *,
        batch_size=1024,
        min_count=0,
        incremental=False,
        src_col="srcip",
        dst_col="dstip",
        dport_col="dsport",
        proto_col="proto",
        use_prefix=True,
        num_workers=0,
    ):
        # 1) vocab / model
        if not incremental:
            self.token2id, self.id2token, freqs = build_vocab_from_df_ip2vec(
                df,
                src_col=src_col, dst_col=dst_col, dport_col=dport_col, proto_col=proto_col,
                min_count=min_count, use_prefix=use_prefix
            )
            vocab_size = len(self.token2id)
            self.unigram_table = torch.tensor(freqs, dtype=torch.float32, device=self.device)

            self.model = SkipGramNegSampling(vocab_size, self.embedding_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            if self.model is None or self.unigram_table is None or not self.token2id or self.optimizer is None:
                raise RuntimeError("incremental=True なら、model/token2id/unigram_table/optimizer が事前に必要です。")

        # 2) dataset / dataloader
        dataset = IP2VecPairDataset(
            df,
            self.token2id,
            src_col=src_col, dst_col=dst_col, dport_col=dport_col, proto_col=proto_col,
            use_prefix=use_prefix,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=ip2vec_collate_fn,
            drop_last=False,
        )

        # 3) train
        assert self.model is not None
        assert self.optimizer is not None
        assert self.unigram_table is not None

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_pairs = 0

            for center_ids, pos_ids in dataloader:
                if center_ids.numel() == 0:
                    continue

                center_ids = center_ids.to(self.device)  # (B_pairs,)
                pos_ids    = pos_ids.to(self.device)     # (B_pairs,)
                B = center_ids.size(0)

                neg_ids = torch.multinomial(
                    self.unigram_table, self.neg_k * B, replacement=True
                ).view(B, self.neg_k).to(self.device)  # ★ deviceに載せる

                loss_vec = self.model(center_ids, pos_ids, neg_ids)
                loss = loss_vec.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # （任意）ノルム制約：必要ならここで射影
                if self.norm_radius is not None:
                    with torch.no_grad():
                        W = self.model.in_embed.weight
                        norms = W.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        factor = torch.clamp(self.norm_radius / norms, max=1.0)
                        W.mul_(factor)

                total_loss += loss.item() * B
                total_pairs += B

            print(f"[epoch {epoch+1}/{self.epochs}] loss={total_loss/max(total_pairs,1):.4f} pairs={total_pairs}")

        self._attach_wv()

    # -----------------------------
    # save / load
    # -----------------------------
    def state_dict(self):
        if self.model is None:
            raise RuntimeError("model is None")
        return {
            "model_state": self.model.state_dict(),
            "token2id": self.token2id,
            "id2token": self.id2token,
            "embedding_size": self.embedding_size,
        }

    def load_state_dict(self, ckpt: dict):
        self.token2id = ckpt["token2id"]
        self.id2token = ckpt["id2token"]
        self.embedding_size = ckpt.get("embedding_size", self.embedding_size)

        self.model = SkipGramNegSampling(len(self.token2id), self.embedding_size).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # unigram_table は ckpt には入れてないので、incrementalするなら外で再構築してセットする
        self._attach_wv()
