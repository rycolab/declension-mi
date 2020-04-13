from .base import BaseData


class CzechData(BaseData):
    file_name = 'czech.final'
    header = None
    sep = '\t'

    @classmethod
    def process_data(cls, df, rare_mode='drop', verbose=True):
        df['gender'] = df[2]

        df['word'] = df[0]
        df['root'] = df[0]
        df['inflection'] = df[1]

        if verbose:
            print('Original number of words: %d. # classes: %d' % (df.shape[0], df['inflection'].unique().shape[0]))
        df.drop_duplicates(subset=['word', 'inflection', 'gender'], inplace=True)
        df = cls.handle_rare_inflections(df, mode=rare_mode, verbose=verbose)

        df['item_id'] = range(df.shape[0])
        return df
