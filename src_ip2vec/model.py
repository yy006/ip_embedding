import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
import numpy as np
import random
import pdb

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
#device = th.device('cpu')

class Skipgram(nn.Module):
    def __init__(self,vocab_size,emb_dim, lamb):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.u_embedding = nn.Embedding(vocab_size,emb_dim)
        self.v_embedding = nn.Embedding(vocab_size,emb_dim)
        self.log_sigmoid = nn.LogSigmoid()

        init_range= 0.5/emb_dim
        self.u_embedding.weight.data.uniform_(-init_range,init_range)
        self.v_embedding.weight.data.uniform_(-0,0)
        self.lamb = lamb

    def forward(self, target, context, neg):
        #pdb.set_trace()
        u_embedd = self.u_embedding(target)
        v_embedd = self.v_embedding(context)
        positive = self.log_sigmoid(th.sum(u_embedd * v_embedd, dim =1)).squeeze()

        v_hat = self.v_embedding(neg)
        #negative_ = th.bmm(u_hat, v_embedd.unsqueeze(2)).squeeze(2)
        negative_ = (v_embedd.unsqueeze(1) * v_hat).sum(2)
        negative = self.log_sigmoid(-th.sum(negative_,dim=1)).squeeze()
        #ここにλΣh-h_0の項の計算を追加
        #h_0 = 1/self.vocab_size*sum([self.u_embedding(th.tensor(i).to(device)) for i in neg[:0]])

        #lossにλΣh-h_0の項を追加
        """
        def pull_center(lamb, target, u_embedding, h_0):
          R= -lamb*sum([th.norm(u_embedding(th.tensor(i).to(device))-h_0)**2 for i in target])
          return R
        """

        #loss = positive + negative + pull_center(self.lamb, target, self.u_embedding, h_0) if self.is_pull_center else positive + negative
        loss = positive + negative
        return -loss.mean()