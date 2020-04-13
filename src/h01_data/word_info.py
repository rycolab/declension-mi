import pickle

from utils.paths import ApplicationPaths
from utils import constants


class WordInfo(object):
    def __init__(self, lang):
        self.lang = lang
        self.model = self.load(lang)

    @classmethod
    def load(cls, lang, rare_mode):
        fpath = cls.get_fpath(lang, rare_mode)
        with open(fpath, 'rb') as f:
            model = pickle.load(f)
        return model

    @classmethod
    def build(cls, df, lang, rare_mode):
        if lang not in constants.controls:
            cls.save({}, lang, rare_mode)
        else:
            cls._build(df, lang, rare_mode)

    @classmethod
    def _build(cls, df, lang, rare_mode):
        info = {}
        for control in constants.controls[lang]:
            info[control] = cls.build_control(df, control)

        cls.save(info, lang, rare_mode)

    @classmethod
    def build_control(cls, df, control):
        control_classes = df[control].unique()
        nclasses = control_classes.shape[0]
        control_map = {x: i for i, x in enumerate(control_classes)}
        controls = {}
        for index, x in df.iterrows():
            controls[x.item_id] = control_map[x[control]]

        info = {
            'controls': controls,
            'nclasses': nclasses,
            'control_map': control_map,
        }
        return info

    @classmethod
    def save(cls, model, lang, rare_mode):
        fpath = cls.get_fpath(lang, rare_mode)
        with open(fpath, 'wb') as f:
            pickle.dump(model, f, protocol=-1)

    @staticmethod
    def get_fpath(lang, rare_mode):
        fname = 'data-%s-%s-word_info.pickl' % (lang, rare_mode)
        fpath = ApplicationPaths.datasets(subfolder_path='preprocess', file_name=fname)
        return fpath
