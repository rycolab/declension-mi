import pickle

from utils.paths import ApplicationPaths


class Word2Vec(object):
    def __init__(self, lang):
        self.lang = lang
        self.model = self.load(lang)

    @classmethod
    def load_not_none(cls, lang, rare_mode):
        model = cls.load(lang, rare_mode)
        return {k: x for k, x in model.items() if x is not None}

    @classmethod
    def load(cls, lang, rare_mode):
        fpath = cls.get_fpath(lang, rare_mode)
        with open(fpath, 'rb') as f:
            model = pickle.load(f)
        return model

    @classmethod
    def save(cls, model, lang, rare_mode):
        fpath = cls.get_fpath(lang, rare_mode)
        with open(fpath, 'wb') as f:
            pickle.dump(model, f, protocol=-1)

    @staticmethod
    def get_fpath(lang, rare_mode):
        fname = 'data-%s-%s-word2vec.pickl' % (lang, rare_mode)
        fpath = ApplicationPaths.datasets(subfolder_path='preprocess', file_name=fname)
        return fpath
