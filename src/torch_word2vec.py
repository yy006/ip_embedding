import math
import random
from collections import Counter

from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def build_pairs_with_flags(ips_seqs, label_seqs, token2id, window):
    pairs = []
    flags = []
    for ips, labs in zip(ips_seqs, label_seqs):
        # 語彙にない IP を落としつつ Label を揃える
        valid = [(token2id[ip], lab) for ip, lab in zip(ips, labs) if ip in token2id]
        if not valid:
            continue
        ids, labs = zip(*valid)  # tuple -> tuple
        ids  = list(ids)
        labs = list(labs)

        L = len(ids)
        for i in range(L):
            c_id = ids[i]
            for j in range(max(0, i-window), min(L, i+window+1)):
                if i == j:
                    continue
                o_id = ids[j]
                pairs.append((c_id, o_id))
                # どちらかが攻撃フローなら 1.0
                flag = float(max(labs[i], labs[j]))
                flags.append(flag)

    pairs_np = np.array(pairs, dtype=np.int64)
    flags_np = np.array(flags, dtype=np.float32)
    return pairs_np, flags_np

class SkipGramNegSampling(nn.Module):
    """
    中身の埋め込みモデル本体 (input, output の2つの埋め込みテーブル)。
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        # 初期化 (word2vec っぽく)
        initrange = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -initrange, initrange)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_ids, pos_ids, neg_ids):
        """
        center_ids: (B,)
        pos_ids   : (B,)
        neg_ids   : (B, K)
        """
        v_c = self.in_embed(center_ids)       # (B, D)
        v_p = self.out_embed(pos_ids)         # (B, D)
        v_n = self.out_embed(neg_ids)         # (B, K, D)

        # 正例: log σ(v_c・v_p)
        pos_score = torch.sum(v_c * v_p, dim=1)          # (B,)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        # 負例: Σ log σ(- v_c・v_nk)
        neg_score = torch.bmm(
            v_n.neg(),                        # (B, K, D)
            v_c.unsqueeze(2)                  # (B, D, 1)
        ).squeeze(2)                          # (B, K)
        neg_loss = torch.log(torch.sigmoid(neg_score) + 1e-10).sum(dim=1)  # (B,)

        # 符号をひっくり返して loss
        loss = -(pos_loss + neg_loss)        # (B,)
        return loss  # サンプルごとの loss（重みを掛けたいので割らない）


class Word2VecDataset(Dataset):
    """
    skip-gram 用に、(center, context) を列挙した Dataset。
    negative sampling は DataLoader 側で行う。
    """
    def __init__(self, pairs, unigram_table, neg_k):
        """
        pairs: list of (center_id, context_id)
        unigram_table: torch.Tensor, 負例サンプル用分布
        neg_k: 負例サンプル数
        """
        self.pairs = pairs
        self.unigram_table = unigram_table
        self.neg_k = neg_k
        self.vocab_size = len(unigram_table)

    def __len__(self):
        return len(self.pairs)
    '''
    def __getitem__(self, idx):
        c, o = self.pairs[idx]
        # 負例サンプル K 個
        neg_ids = torch.multinomial(self.unigram_table, self.neg_k, replacement=True)
        return torch.LongTensor([c]), torch.LongTensor([o]), neg_ids
    '''
    def __getitem__(self, idx):
        # ここでは negative sampling をしない
        c, o = self.pairs[idx]
        return torch.LongTensor([c]), torch.LongTensor([o])


class TorchWord2Vec:
    """
    あなたの gensim Word2Vec ラッパークラスを、PyTorch で書き直した版。
    - skip-gram + negative sampling
    - anomaly_scores による重み付き loss (レベル3の入口)
    """
    def __init__(
        self,
        c=25,
        e=50,
        epochs=2,
        mname="sample",
        neg_k=5,
        lr=0.025,
        device=None,
        method=None,
        alpha_anom=0.0,            # 異常重みの強さ (0なら通常のSGNS)
        contrastive_lambda=0.0,    # 将来、contrastive lossを足す用
    ):
        self.context_window = c
        self.embedding_size = e
        self.epochs = epochs
        self.mname = mname
        self.neg_k = neg_k
        self.lr = lr
        self.alpha_anom = alpha_anom
        self.contrastive_lambda = contrastive_lambda

        self.token2id = {}
        self.id2token = {}
        self.model = None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class _WVWrapper:
        """
        gensim の KeyedVectors っぽいインターフェースを PyTorch 上でエミュレートするクラス。
        - index_to_key
        - vector_size
        - __getitem__(token or index)
        などを提供する。
        """
        def __init__(self, parent: "TorchWord2Vec"):
            self._parent = parent

        @property
        def index_to_key(self):
            # gensim の index_to_key は「単語リスト」なので、とりあえず token2id のキー順で返す
            # 必要なら頻度順ソートにしてもよい
            return list(self._parent.token2id.keys())

        @property
        def vector_size(self):
            return self._parent.embedding_size

        def __getitem__(self, key):
            """
            wv['10.0.0.1'] or wv[123] に対応。
            """
            parent = self._parent
            if isinstance(key, str):
                idx = parent.token2id.get(key)
                if idx is None:
                    raise KeyError(f"Unknown token: {key}")
            else:
                # int index の場合
                idx = int(key)
                if idx < 0 or idx >= len(parent.id2token):
                    raise IndexError(f"Index out of range: {idx}")

            with torch.no_grad():
                vec = parent.model.in_embed.weight[idx].detach().cpu().numpy()
            return vec

        def __repr__(self):
            return (f"PyTorchWVWrapper(size={len(self.index_to_key)}, "
                    f"vector_size={self.vector_size})")

    def _attach_wv(self):
        """
        self.model.wv に gensim 風のラッパを差し込む。
        """
        if self.model is None:
            return
        self.model.wv = TorchWord2Vec._WVWrapper(self)

    # ====== 語彙構築 & コーパス前処理 ======
    def _build_vocab(self, corpus, min_count=0):
        """
        corpus: list[list[str]] の想定
        """
        counter = Counter()
        for sent in corpus:
            counter.update(sent)

        # min_count 以上のトークンを語彙に載せる
        tokens = [t for t, f in counter.items() if f >= min_count]
        tokens = sorted(tokens)
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

        # 負例サンプル用の unigram^(3/4) 分布
        freqs = np.array([counter[t] for t in tokens], dtype=np.float64)
        freqs = freqs ** 0.75
        freqs = freqs / freqs.sum()
        self.unigram_table = torch.tensor(freqs, dtype=torch.float32)

    def _corpus_to_pairs(self, corpus):
        """
        window 幅 self.context_window で skip-gram 用の (center, context) ペアを作る。
        """
        pairs = []
        for sent in corpus:
            ids = [self.token2id[t] for t in sent if t in self.token2id]
            for i, center in enumerate(ids):
                left = max(0, i - self.context_window)
                right = min(len(ids), i + self.context_window + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    context = ids[j]
                    pairs.append((center, context))
        return pairs

    '''
    # ====== 学習 ======
    def train(self, corpus, anomaly_scores=None, batch_size=1024, min_count=0, save=False):
        """
        corpus: list[list[str]]
        anomaly_scores: dict[token -> score(0〜1)] など。
            (異常に近いほどスコアを大きくしておく)
            loss に (1 + alpha_anom * score_pair) を掛ける。
        """
        # 1. vocab 構築
        self._build_vocab(corpus, min_count=min_count)
        vocab_size = len(self.token2id)

        # 2. (center, context) ペア作成
        pairs = self._corpus_to_pairs(corpus)

        # 3. Dataset / DataLoader
        dataset = Word2VecDataset(
            pairs=pairs,
            unigram_table=self.unigram_table,
            neg_k=self.neg_k,
        )

        # pairsを表示
        print(len(pairs), "pairs generated.")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=cpu_count())

        # 4. モデル & optimizer 用意
        self.model = SkipGramNegSampling(vocab_size, self.embedding_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 5. 学習ループ
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for center_ids, pos_ids in dataloader:
                center_ids = center_ids.squeeze(1).to(self.device)  # (B,)
                pos_ids = pos_ids.squeeze(1).to(self.device)        # (B,)
                # neg_ids = neg_ids.to(self.device)                   # (B, K)

                B = center_ids.size(0)

                # 一発で負例サンプリング
                neg_ids = torch.multinomial(
                    self.unigram_table,
                    B * self.neg_k,
                    replacement=True
                ).view(B, self.neg_k).to(self.device)  # (B, K)

                # SGNS のサンプルごと loss (B,)
                loss_vec = self.model(center_ids, pos_ids, neg_ids)

                # === レベル3: 異常重みを掛ける部分 ===
                if anomaly_scores is not None and self.alpha_anom > 0.0:
                    # center / context のスコアを取得 (なければ0)
                    c_scores = torch.tensor(
                        [anomaly_scores.get(self.id2token[int(i)], 0.0) for i in center_ids.cpu()],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    o_scores = torch.tensor(
                        [anomaly_scores.get(self.id2token[int(i)], 0.0) for i in pos_ids.cpu()],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    pair_scores = 0.5 * (c_scores + o_scores)  # 平均
                    weights = 1.0 + self.alpha_anom * pair_scores  # (B,)
                    loss_vec = loss_vec * weights

                # === ここに contrastive loss 等を足す余地あり ===
                # if self.contrastive_lambda > 0:
                #     contrastive_loss = ... (IPラベルや時間窓情報から作る)
                #     loss_vec = loss_vec + self.contrastive_lambda * contrastive_loss_per_sample

                loss = loss_vec.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * center_ids.size(0)

            avg_loss = total_loss / len(dataset)
            print(f"[epoch {epoch+1}/{self.epochs}] loss={avg_loss:.4f}")

        self._attach_wv()
        if save:
            self.save_model()
    '''
    def train(self, corpus, anomaly_scores=None, batch_size=1024, min_count=0, save=False):
        """
        corpus: list[list[str]]
            トークン列（IPアドレスなど）のリスト。各要素が1文（1フロー列）に対応。
        anomaly_scores: dict[token -> score(0〜1)] など（任意）
            各トークンに対して「どれくらい異常寄りか」を表すスコア。
            0: まったく異常でない、1: 強く異常、といったイメージ。
            損失計算時に (1 + alpha_anom * score_pair) の重みとして用いる。
        batch_size: int
            ミニバッチサイズ。pairs（center, context）の数に対して 1024 前後を目安に調整。
        min_count: int
            語彙に載せる最小出現回数。これ未満のトークンはUNK的に落とす。
        save: bool
            True の場合、学習終了後に self.save_model() で PyTorch モデルを保存する。
        """

        # 1. 語彙構築（コーパスから token2id, id2token, unigram_table を作る）
        self._build_vocab(corpus, min_count=min_count)
        vocab_size = len(self.token2id)

        # 2. (center, context) ペア作成
        #    - window サイズ self.context_window に基づいて skip-gram のペアを全列挙
        #    - ここでは一度すべてメモリに載せて numpy 配列に変換しておく
        pairs = self._corpus_to_pairs(corpus)
        num_pairs = len(pairs)
        print(num_pairs, "pairs generated.")

        centers_np = np.array([c for (c, _) in pairs], dtype=np.int64)
        contexts_np = np.array([o for (_, o) in pairs], dtype=np.int64)

        # 3. モデル & optimizer 準備
        #    - SkipGramNegSampling: center / context / negative の3種類の埋め込みから
        #      SGNS の損失を計算するモジュール
        #    - optimizer は埋め込みに適した SparseAdam を想定
        self.model = SkipGramNegSampling(vocab_size, self.embedding_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 4. 学習ループ
        #    - 各 epoch ごとに全 pair をシャッフルし、
        #      自前でミニバッチ（start:end）を切って学習する。
        self.model.train()
        for epoch in range(self.epochs):
            # 4-0. ペアのインデックスをランダムシャッフル
            perm = np.random.permutation(num_pairs)

            total_loss = 0.0
            for start in range(0, num_pairs, batch_size):
                end = min(start + batch_size, num_pairs)
                idx = perm[start:end]          # このバッチで使う pair のインデックス
                B = end - start                # 実際のバッチサイズ（末尾は batch_size 未満になる）

                # 4-1. numpy → Tensor に変換して GPU / CPU に載せる
                center_ids = torch.from_numpy(centers_np[idx]).to(self.device)   # (B,)
                pos_ids    = torch.from_numpy(contexts_np[idx]).to(self.device)  # (B,)

                # 4-2. 負例サンプリングをバッチ単位で一括生成
                #      - unigram^(3/4) 分布 self.unigram_table から
                #        self.neg_k * B 個まとめて引いて (B, K) に reshape
                neg_ids = torch.multinomial(
                    self.unigram_table,
                    self.neg_k * B,
                    replacement=True,
                ).view(B, self.neg_k).to(self.device)                             # (B, K)

                # 4-3. SGNS のサンプルごとの loss ベクトル (B,) を計算
                loss_vec = self.model(center_ids, pos_ids, neg_ids)

                # 4-4. レベル3: 異常スコアに基づく重み付け（任意）
                #      - anomaly_scores が与えられ、alpha_anom > 0 のときだけ有効
                #      - center / pos のトークンそれぞれからスコアを取り、
                #        ペアの平均スコアで重み (1 + alpha_anom * score) を掛ける。
                if anomaly_scores is not None and self.alpha_anom > 0.0:
                    # center / context のスコアを取得（存在しないトークンは 0 とみなす）
                    c_scores = torch.tensor(
                        [anomaly_scores.get(self.id2token[int(i)], 0.0) for i in center_ids.cpu()],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    o_scores = torch.tensor(
                        [anomaly_scores.get(self.id2token[int(i)], 0.0) for i in pos_ids.cpu()],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    pair_scores = 0.5 * (c_scores + o_scores)           # (B,)
                    weights = 1.0 + self.alpha_anom * pair_scores       # (B,)
                    loss_vec = loss_vec * weights

                # 4-5. contrastive loss などを将来追加する場合はここで loss_vec に足す
                # if self.contrastive_lambda > 0:
                #     contrastive_loss_per_sample = ...
                #     loss_vec = loss_vec + self.contrastive_lambda * contrastive_loss_per_sample

                # 4-6. ミニバッチ平均の loss を計算して逆伝播
                loss = loss_vec.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * B

            # 4-7. epoch 全体の平均 loss をログ出力
            avg_loss = total_loss / num_pairs
            print(f"[epoch {epoch+1}/{self.epochs}] loss={avg_loss:.4f}")

        # 5. gensim 互換インタフェース (model.wv) をアタッチ
        self._attach_wv()

        # 6. モデル保存（必要な場合のみ）
        if save:
            self.save_model()

    def train_from_pairs(self, pairs, pair_flags, batch_size=1024, save=False):
        centers_np = pairs[:, 0].astype(np.int64)
        contexts_np = pairs[:, 1].astype(np.int64)
        flags_np    = pair_flags.astype(np.float32)

        num_pairs = len(pairs)

        self.model = SkipGramNegSampling(vocab_size=len(self.token2id),
                                        embedding_dim=self.embedding_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            perm = np.random.permutation(num_pairs)
            total_loss = 0.0
            for start in range(0, num_pairs, batch_size):
                end = min(start+batch_size, num_pairs)
                idx = perm[start:end]
                B = end - start

                center_ids = torch.from_numpy(centers_np[idx]).to(self.device)
                pos_ids    = torch.from_numpy(contexts_np[idx]).to(self.device)
                neg_ids = torch.multinomial(
                    self.unigram_table,
                    self.neg_k * B,
                    replacement=True,
                ).view(B, self.neg_k).to(self.device)

                loss_vec = self.model(center_ids, pos_ids, neg_ids)

                if self.alpha_anom > 0.0:
                    #print("|||||Applying anomaly weights in train_from_pairs|||||")
                    flags = torch.from_numpy(flags_np[idx]).to(self.device)  # 0 or 1
                    weights = 1.0 + self.alpha_anom * flags
                    loss_vec = loss_vec * weights

                loss = loss_vec.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * B

            print(f"[epoch {epoch+1}/{self.epochs}] loss={total_loss/num_pairs:.4f}")

        self._attach_wv()
        if save:
            self.save_model()

    def train_with_labels(
        self,
        ips_seqs,
        label_seqs,
        batch_size=1024,
        min_count=0,
        save=False,
    ):
        """
        ips_seqs   : list[list[str]]  各サービス/時間窓ごとの IP 列
        label_seqs : list[list[int]]  同じ長さの攻撃ラベル列 (0/1)
        """
        # 1) 語彙構築（IP だけ使う）
        self._build_vocab(ips_seqs, min_count=min_count)

        # 2) ラベル付きペア＋フラグ作成
        pairs_np, flags_np = build_pairs_with_flags(
            ips_seqs,
            label_seqs,
            token2id=self.token2id,
            window=self.context_window,
        )
        print(f"{len(pairs_np)} labeled pairs generated.")

        # 3) そのまま train_from_pairs に渡して学習
        self.train_from_pairs(
            pairs=pairs_np,
            pair_flags=flags_np,
            batch_size=batch_size,
            save=save,
        )

    # ====== モデルの保存・読み込み ======
    def save_model(self, path=None):
        path = path or f"{self.mname}_torchw2v.pt"
        state = {
            "model_state": self.model.state_dict(),
            "token2id": self.token2id,
            "id2token": self.id2token,
            "config": {
                "c": self.context_window,
                "e": self.embedding_size,
                "epochs": self.epochs,
                "mname": self.mname,
                "neg_k": self.neg_k,
                "lr": self.lr,
                "alpha_anom": self.alpha_anom,
                "contrastive_lambda": self.contrastive_lambda,
            },
        }
        torch.save(state, path)

    def load_model(self, path=None):
        path = path or f"{self.mname}_torchw2v.pt"
        state = torch.load(path, map_location=self.device)
        self.token2id = state["token2id"]
        self.id2token = state["id2token"]
        cfg = state["config"]
        self.context_window = cfg["c"]
        self.embedding_size = cfg["e"]
        self.epochs = cfg["epochs"]
        self.mname = cfg["mname"]
        self.neg_k = cfg["neg_k"]
        self.lr = cfg["lr"]
        self.alpha_anom = cfg["alpha_anom"]
        self.contrastive_lambda = cfg["contrastive_lambda"]

        vocab_size = len(self.token2id)
        self.model = SkipGramNegSampling(vocab_size, self.embedding_size).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

        self._attach_wv()

    # ====== 埋め込みの取得 ======
    def get_embeddings(self, ips, labels=None):
        """
        gensim版と似たインターフェース。
        ips: list[str] (token名前提。IPアドレスをそのままtokenにしてもOK)
        labels: ip/class を持つ DataFrame (任意)
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")

        self.model.eval()
        with torch.no_grad():
            embs = []
            for t in ips:
                if t not in self.token2id:
                    # 未知語の場合はゼロベクトルなど適当に
                    embs.append(np.zeros(self.embedding_size, dtype=np.float32))
                else:
                    idx = self.token2id[t]
                    vec = self.model.in_embed.weight[idx].cpu().numpy()
                    embs.append(vec)

        df = pd.DataFrame(embs, index=ips)
        df = df.reset_index().rename(columns={"index": "ip"})

        if labels is not None:
            df = df.merge(labels, on="ip", how="left").set_index("ip")
        else:
            df = df.set_index("ip")

        return df

    def __repr__(self):
        if self.model is not None:
            vocab = len(self.token2id)
            vsize = self.embedding_size
            return (f"TorchWord2Vec(model=PyTorch, mname={self.mname}, "
                    f"vector_size={vsize}, window={self.context_window}, "
                    f"epochs={self.epochs}, vocab_size={vocab})")
        else:
            return (f"TorchWord2Vec(model=None, mname={self.mname}, "
                    f"embedding_size={self.embedding_size}, "
                    f"window={self.context_window}, epochs={self.epochs})")
