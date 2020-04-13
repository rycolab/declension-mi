import copy
import torch
import torch.nn as nn

from .context import EmbeddingContext, BaseContext, Word2VecContext, ControlWord2VecContext
from utils import constants


class BaseLM(nn.Module):
    name = 'base'

    def __init__(self, vocab_size, n_classes, hidden_size, lang, rare_mode, nlayers=1, dropout=0.1, embedding_size=None,
                 word2vec_size=10, context=None, controls=None):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.word2vec_size = word2vec_size
        self.dropout_p = dropout
        self.vocab_size = vocab_size
        self.n_classes = n_classes

        self.best_state_dict = None
        self.lang = lang
        self.rare_mode = rare_mode
        self.load_context(context, controls)

    def load_context(self, context, controls):
        self.context_type = context
        self.controls = controls
        if context is None or context == 'none':
            if self.lang in constants.controls:
                self.context = EmbeddingContext(self.hidden_size, self.lang, self.rare_mode, nlayers=self.nlayers)
            else:
                self.context = BaseContext(self.hidden_size, nlayers=self.nlayers)
        elif context == 'word2vec':
            if self.lang in constants.controls:
                self.context = ControlWord2VecContext(
                    self.hidden_size, self.lang, self.rare_mode, nlayers=self.nlayers, pca_size=self.word2vec_size)
            else:
                self.context = Word2VecContext(
                    self.hidden_size, self.lang, self.rare_mode, nlayers=self.nlayers, pca_size=self.word2vec_size)
        else:
            raise ValueError('Invalid context name %s' % context)

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path, context, suffix):
        fname = self.get_name(path, context, suffix)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'vocab_size': self.vocab_size,
            'context': self.context_type,
            'controls': self.controls,
            'lang': self.lang,
            'rare_mode': self.rare_mode,
        }

    @classmethod
    def load(cls, path, suffix):
        checkpoints = cls.load_checkpoint(path, suffix)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path, context, suffix):
        fname = cls.get_name(path, context, suffix)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path, context, suffix):
        return '%s/%s__%s__%s.tch' % (path, cls.name, context, suffix)
