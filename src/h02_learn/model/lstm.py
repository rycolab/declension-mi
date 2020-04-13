import copy
import torch
import torch.nn as nn

from .base import BaseLM


class LstmLM(BaseLM):
    name = 'lstm'

    def __init__(self, vocab_size, n_classes, hidden_size, lang, rare_mode,
                 nlayers=1, dropout=0.1, embedding_size=None, **kwargs):
        super().__init__(vocab_size, n_classes, hidden_size, lang, rare_mode, nlayers=nlayers, dropout=dropout,
                         embedding_size=embedding_size, **kwargs)

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, hidden_size, nlayers,
                            dropout=(dropout if nlayers > 1 else 0), batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, n_classes)

    def forward(self, x, idx):
        h_old = self.context(idx)
        x_emb = self.dropout(self.get_embedding(x))
        c_t, h_t = self.lstm(x_emb, h_old)

        c_t = self.get_last_output(c_t, x)
        c_t = self.dropout(c_t).contiguous()

        logits = self.out(c_t)
        return logits

    def get_last_output(self, c, x):
        # get lengths ignoring padding
        lengths = (x != 0).sum(1)

        idx = (lengths - 1).view(-1, 1).expand(
            lengths.shape[0], c.size(2))
        idx = idx.unsqueeze(1)
        # ToDo: think if this is the best strategy to get representation
        return c.gather(1, idx).squeeze(1)

    def get_embedding(self, x):
        return self.embedding(x)
