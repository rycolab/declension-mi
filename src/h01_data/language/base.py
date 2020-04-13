import pandas as pd

import sys
sys.path.append('./')
from utils.paths import ApplicationPaths


class BaseData(object):
    keep_default_na = True
    subfolder_path = ''

    def __init__(self, lang, rare_mode='drop'):
        self.lang = lang
        self.rare_mode = rare_mode
        self.data, self.inflection_labels = \
            self.get_data(lang, rare_mode=rare_mode)

    @classmethod
    def get_data(cls, lang, rare_mode='drop', verbose=True):
        df = cls.get_main_data(lang, rare_mode=rare_mode, verbose=verbose)
        return cls.label_inflections(df)

    @classmethod
    def get_main_data(cls, lang, rare_mode='drop', verbose=True):
        df = cls.read_raw_data()
        return cls.process_data(df, rare_mode=rare_mode, verbose=verbose)

    @classmethod
    def read_raw_data(cls):
        file = ApplicationPaths.datasets(file_name=cls.file_name, subfolder_path=cls.subfolder_path)
        return pd.read_csv(file, sep=cls.sep, header=cls.header, keep_default_na=cls.keep_default_na)

    @classmethod
    def handle_rare_inflections(cls, df, min_occurrences=20, column=0, mode='drop', verbose=True):
        rare_inflections = cls.get_rare_inflections(
            df, min_occurrences=min_occurrences, column=column, verbose=verbose)
        if mode == 'drop':
            return cls.drop_rare_inflections(df, rare_inflections)
        elif mode == 'group':
            return cls.group_rare_inflections(df, rare_inflections)
        else:
            raise ValueError('Invalid value for mode to handle rare inflections: %s' % mode)

    @staticmethod
    def group_rare_inflections(df, rare_inflections):
        df.loc[df[df.inflection.isin(rare_inflections)].index, 'inflection'] = 'irregular'
        return df

    @staticmethod
    def drop_rare_inflections(df, rare_inflections):
        return df[~df.inflection.isin(rare_inflections)].copy()

    @staticmethod
    def get_rare_inflections(df, min_occurrences=20, column=0, verbose=True):
        df_grouped = df.groupby('inflection').agg('count')[column]
        rare_items = df_grouped <= min_occurrences
        if verbose:
            print('Found %d items with rare inflections from %d classes' %
                  ((df_grouped[rare_items]).sum(), (rare_items).sum()))

        df_grouped = df_grouped[rare_items]
        return df_grouped.index.unique()

    @classmethod
    def label_inflections(cls, df):
        inflections = df['inflection'].unique()
        inflection_labels = {x: i for i, x in enumerate(inflections)}

        df['inflection_label'] = df['inflection'].apply(lambda x: inflection_labels[x])
        df = df.sample(frac=1).reset_index(drop=True)

        return df, inflection_labels
