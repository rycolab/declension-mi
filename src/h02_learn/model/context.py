import torch
import torch.nn as nn
import pickle
import math
import numpy as np
from sklearn.decomposition import PCA

from h01_data.word_info import WordInfo
from h01_data.word2vec import Word2Vec


class Context(nn.Module):
    def __init__(self, hidden_size, nlayers=1):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size


class BaseContext(Context):
    def __init__(self, hidden_size, nlayers=1):
        super().__init__(hidden_size, nlayers=nlayers)
        self.get_const_embs(hidden_size, nlayers)

    def get_const_embs(self, hidden_size, nlayers):
        hidden_tensor = torch.Tensor(self.nlayers, 1, self.hidden_size).uniform_(-0.01, 0.01)
        self.init_c = nn.Parameter(hidden_tensor)
        hidden_tensor = torch.Tensor(self.nlayers, 1, self.hidden_size).uniform_(-0.01, 0.01)
        self.init_h = nn.Parameter(hidden_tensor)

    def forward(self, x):
        bsz = x.size(0)
        return self.init_c.repeat(1, bsz, 1), self.init_h.repeat(1, bsz, 1)


class MultiEmbedding(nn.Module):
    def __init__(self, named_vocab_sizes, hidden_size, nlayers=1):
        super().__init__()
        self.vocab_sizes = named_vocab_sizes
        self.hidden_size = hidden_size
        self.hidden_size_each = int(hidden_size / len(self.vocab_sizes))
        self.nlayers = nlayers
        self.get_embs(nlayers)

    def get_embs(self, nlayers):
        self.c_embeddings, self.h_embeddings = [], []
        iter_sizes = [x for x in self.vocab_sizes.values()]
        for vocab_size in iter_sizes[:-1]:
            self.c_embeddings += [nn.Embedding(vocab_size, self.hidden_size_each * nlayers)]
            self.h_embeddings += [nn.Embedding(vocab_size, self.hidden_size_each * nlayers)]

        hidden_size_last = self.hidden_size - (self.hidden_size_each * (len(iter_sizes) - 1))
        self.c_embeddings += [nn.Embedding(iter_sizes[-1], hidden_size_last * nlayers)]
        self.h_embeddings += [nn.Embedding(iter_sizes[-1], hidden_size_last * nlayers)]

        self.c_embeddings = nn.ModuleList(self.c_embeddings)
        self.h_embeddings = nn.ModuleList(self.h_embeddings)

    def forward(self, idxs):
        bsz = idxs[0].size(0)

        cs = [self.c_embeddings[i](idx).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous()
              for i, idx in enumerate(idxs)]
        hs = [self.h_embeddings[i](idx).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous()
              for i, idx in enumerate(idxs)]

        cs = torch.cat(cs, dim=-1)
        hs = torch.cat(hs, dim=-1)

        return cs, hs


class EmbeddingContext(Context):
    def __init__(self, hidden_size, lang, rare_mode, nlayers=1, dropout=0.1):
        super().__init__(hidden_size, nlayers=nlayers)
        self.lang = lang
        self.rare_mode = rare_mode
        self.dropout = nn.Dropout(dropout)
        self.get_embs(hidden_size, nlayers, lang, rare_mode)

    def get_embs(self, hidden_size, nlayers, lang, rare_mode):
        self.controls = WordInfo.load(lang, rare_mode)
        self.ncontrols = len(self.controls)

        embs_sizes = {control_type: control['nclasses'] for control_type, control in self.controls.items()}
        self.embeddings = MultiEmbedding(embs_sizes, hidden_size, nlayers)

        self.id_to_pos = []
        for control_type, control in self.controls.items():
            vec = self.dict_to_vec(control['controls'])
            vec_tensor = torch.Tensor(vec).long()
            self.id_to_pos += [nn.Parameter(vec_tensor, requires_grad=False)]
        self.id_to_pos = nn.ParameterList(self.id_to_pos)

    @staticmethod
    def dict_to_vec(id_dict):
        max_idx = max(id_dict.keys())
        vec = np.ones(max_idx + 1) * -1
        vec[list(id_dict.keys())] = list(id_dict.values())
        return vec

    def forward(self, x):
        x_pos = [map[x] for map in self.id_to_pos]
        return self.embeddings(x_pos)


class Word2VecContext(Context):
    def __init__(self, hidden_size, lang, rare_mode, nlayers=1, pca_size=10, dropout=0.1):
        super().__init__(hidden_size, nlayers=nlayers)
        self.lang = lang
        self.pca_size = pca_size

        self.build_word2vec_embedding(lang, rare_mode)
        # self.mlp = self.build_mlp()
        self.out = nn.Linear(self.pca_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def build_word2vec_embedding(self, lang, rare_mode):
        self.word2vec = Word2Vec.load_not_none(lang, rare_mode)
        self.word2vec_size = next(iter(self.word2vec.values())).shape[0]
        self.max_id = max(self.word2vec.keys())
        word2vec_pca = self.pca_word2vec_embedding()
        self._build_word2vec_embedding(word2vec_pca)

    def pca_word2vec_embedding(self):
        word2vec_np = np.zeros((len(self.word2vec), self.word2vec_size))
        w2v_key_to_idx = {}
        for i, (key, vec) in enumerate(self.word2vec.items()):
            word2vec_np[i] = vec
            w2v_key_to_idx[key] = i

        pca = PCA(n_components=self.pca_size)
        pca_vecs = pca.fit_transform(word2vec_np)

        word2vec_pca = np.zeros((self.max_id + 1, self.pca_size))
        for key in self.word2vec.keys():
            word2vec_pca[key] = pca_vecs[w2v_key_to_idx[key]]

        return word2vec_pca

    def _build_word2vec_embedding(self, word2vec_pca):
        self.c_embedding = nn.Embedding(word2vec_pca.shape[0], word2vec_pca.shape[1])
        self.c_embedding.weight.data.copy_(nn.Parameter(torch.from_numpy(word2vec_pca), requires_grad=False))
        self.c_embedding.weight.requires_grad = False
        self.h_embedding = nn.Embedding(word2vec_pca.shape[0], word2vec_pca.shape[1])
        self.h_embedding.weight.data.copy_(nn.Parameter(torch.from_numpy(word2vec_pca), requires_grad=False))
        self.h_embedding.weight.requires_grad = False

        self.c_linear = nn.Linear(word2vec_pca.shape[1], self.hidden_size * self.nlayers)
        self.h_linear = nn.Linear(word2vec_pca.shape[1], self.hidden_size * self.nlayers)

    def forward(self, x):
        return self._w2v_forward(x)

    def _w2v_forward(self, x):
        bsz = x.size(0)

        x_c_emb = self.dropout(self.c_embedding(x))
        x_c_emb = self.dropout(
            self.c_linear(x_c_emb).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous())
        x_h_emb = self.dropout(self.h_embedding(x))
        x_h_emb = self.dropout(
            self.h_linear(x_h_emb).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous())

        return x_c_emb, x_h_emb


class ControlWord2VecContext(Context):
    def __init__(self, hidden_size, lang, rare_mode, nlayers=1, pca_size=10, dropout=0.1):
        super().__init__(hidden_size, nlayers=nlayers)
        self.control = EmbeddingContext(
            math.ceil(hidden_size / 2), lang, rare_mode, nlayers=nlayers, dropout=dropout)
        self.w2v = Word2VecContext(
            math.floor(hidden_size / 2), lang, rare_mode, nlayers=nlayers, pca_size=pca_size, dropout=dropout)

    def forward(self, x):
        control_c_embs, control_h_embs = self.control(x)
        w2v_c_embs, w2v_h_embs = self.w2v(x)

        x_c_embs = torch.cat([control_c_embs, w2v_c_embs], dim=-1)
        x_h_embs = torch.cat([control_h_embs, w2v_h_embs], dim=-1)

        return x_c_embs, x_h_embs
