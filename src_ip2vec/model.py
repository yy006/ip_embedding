import torch
import torch.nn as nn

class SkipGramNegSampling(nn.Module):
    """
    SGNS: log σ(u_c·v_p) + Σ log σ(-u_c·v_n)
    - in_embed: u (target/center)
    - out_embed: v (context)
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        initrange = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -initrange, initrange)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_ids, pos_ids, neg_ids):
        """
        center_ids: (B,)
        pos_ids   : (B,)
        neg_ids   : (B, K)
        return    : (B,) sample-wise loss
        """
        u = self.in_embed(center_ids)       # (B, D)
        v_pos = self.out_embed(pos_ids)     # (B, D)
        v_neg = self.out_embed(neg_ids)     # (B, K, D)

        # positive: log σ(u·v_pos)
        pos_score = torch.sum(u * v_pos, dim=1)                 # (B,)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)  # (B,)

        # negative: Σ log σ(-u·v_neg)
        # u: (B,D) -> (B,1,D), v_neg: (B,K,D) => dot along D => (B,K)
        neg_score = torch.sum(v_neg * u.unsqueeze(1), dim=2)    # (B, K)
        neg_loss = torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)  # (B,)

        loss = -(pos_loss + neg_loss)  # (B,)
        return loss.mean()
