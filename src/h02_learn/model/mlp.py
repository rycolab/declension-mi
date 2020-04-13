import numpy as np
import torch
import torch.nn as nn

from .base import BaseLM
from h01_data.word2vec import Word2Vec


class Word2VecMLP(BaseLM):
    name = 'mlp-word2vec'

    def __init__(self, vocab_size, n_classes, hidden_size, lang, rare_mode,
                 nlayers=1, dropout=0.1, **kwargs):
        super().__init__(vocab_size, n_classes, hidden_size, lang, rare_mode,
                         nlayers=nlayers, dropout=dropout, **kwargs)

        self.build_word2vec_embedding(lang, rare_mode)
        self.mlp = self.build_mlp()
        self.out = nn.Linear(self.final_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def build_word2vec_embedding(self, lang, rare_mode):
        self.word2vec = Word2Vec.load_not_none(lang, rare_mode)
        self.word2vec_size = next(iter(self.word2vec.values())).shape[0]
        self.max_id = max(self.word2vec.keys())
        self.embedding = nn.Embedding(self.max_id + 1, self.word2vec_size)
        self.fill_word2vec_embedding()

    def fill_word2vec_embedding(self):
        word2vec = np.zeros((self.max_id + 1, self.word2vec_size))
        for k, vec in self.word2vec.items():
            word2vec[k] = vec
        self.embedding.weight.data.copy_(nn.Parameter(torch.from_numpy(word2vec), requires_grad=False))
        self.embedding.weight.requires_grad = False

    def build_mlp(self):
        src_size = self.word2vec_size + self.hidden_size
        tgt_size = self.hidden_size
        mlp = []
        for layer in range(self.nlayers):
            mlp += [nn.Linear(src_size, tgt_size)]
            mlp += [nn.ReLU()]
            mlp += [nn.Dropout(self.dropout_p)]
            src_size, tgt_size = tgt_size, int(tgt_size / 2)
        self.final_hidden_size = src_size
        return nn.Sequential(*mlp)

    def forward(self, x, idx):
        context = self.context(idx)[0][0]
        x_emb = self.dropout(self.embedding(idx))
        x_full = torch.cat([x_emb, context], dim=-1)
        x = self.mlp(x_full)
        logits = self.out(x)
        return logits
