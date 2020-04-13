from .base import BaseData


class GermanNounsData(BaseData):
    file_name = 'celexSMORNounsClean.tsv'
    header = 'infer'
    sep = '\t'
    keep_default_na = False
    subfolder_path = ''

    equivalencies = [
        ('S1', 'S4'),
        ('P1', 'P7'),
        ('P3', 'P6'),
        ('P8', 'P9'),
        ('P8', 'P10'),
        ('loan', 'P8'),
        ('P1', 'P2'),
        ('P1U', 'P2U'),
    ]

    @classmethod
    def process_data(cls, df, rare_mode='drop', verbose=True):
        cls.preprocess_inflections(df)

        df['gender'] = df['Gender']

        df['word'] = df['Word'].apply(lambda x: x.lower())
        df['root'] = df['Word'].apply(lambda x: x.lower())
        if verbose:
            print('Original number of words: %d. # classes: %d' % (df.shape[0], df['inflection'].unique().shape[0]))
        df.drop_duplicates(subset=['word', 'inflection', 'gender'], inplace=True)
        df = cls.handle_rare_inflections(df, column='Word', mode=rare_mode, verbose=verbose)

        df['item_id'] = range(df.shape[0])
        return df

    @classmethod
    def preprocess_inflections(cls, df):
        df = cls.split_inflections(df)
        cls.check_sanity_inflections(df)

        for x, y in cls.equivalencies:
            df['inflection_singular'] = df['inflection_singular'].apply(lambda label: x if label == y else label)
            df['inflection_plural'] = df['inflection_plural'].apply(lambda label: x if label == y else label)

        df = cls.merge_inflections(df)
        del df['inflection_singular']
        del df['inflection_plural']

    @classmethod
    def split_inflections(cls, df):
        df['inflection_singular'] = df['DeclensionCELEX'].apply(lambda x: x.split('/')[0])
        df['inflection_plural'] = df['DeclensionCELEX'].apply(lambda x: x.split('/')[1])
        return df

    @classmethod
    def check_sanity_inflections(cls, df):
        df['inflection_label_size'] = df['DeclensionCELEX'].apply(lambda x: len(x.split('/')))
        assert (df.inflection_label_size == 2).all(), 'All declension class should have exactly two splits'
        del df['inflection_label_size']

    @classmethod
    def merge_inflections(cls, df):
        df['inflection'] = df.apply(lambda x: '%s/%s' % (x.inflection_singular, x.inflection_plural), axis=1)
        return df
