import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.word2vec import Word2Vec
from h01_data.word_info import WordInfo
from h01_data.language import CzechData, GermanNounsData
from utils.paths import ApplicationPaths
from utils.utils import config_seed
from utils import constants

data_class = {
    'cze': CzechData,
    'ger_n': GermanNounsData,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='cze',
                        help='Language to be parsed')
    parser.add_argument('--rare-mode', type=str, default='', choices=['', 'drop', 'group'],
                        help='Language to be parsed')
    parser.add_argument('-seed', default=123, type=int)
    return parser.parse_args()


def get_data(lang, rare_mode, verbose=True):
    return data_class[lang].get_data(lang, rare_mode, verbose=verbose)


def filter_word2vec(df, lang, rare_mode):
    model = Word2Vec.load_not_none(lang, rare_mode)
    n_dropped = (~df.item_id.isin(model.keys())).sum()
    df = df[df.item_id.isin(model.keys())]
    print('Filtered %d items not in word2vec' % n_dropped)
    return df


def print_basic_stats(df):
    inflections = df['inflection'].unique()
    for infl in inflections:
        n_items = df[df.inflection == infl].shape[0]
        print('Dataset contains %d unique verbs with \'%s\' inflection' % (n_items, infl))


def get_token_map(df):
    tokens = get_tokens(df)
    tokens = sorted(list(tokens))
    token_map = _get_token_map(tokens)

    return token_map


def get_tokens(df):
    tokens = set()
    for _, x in df.iterrows():
        tokens |= set(x['root'])
    return tokens


def _get_token_map(tokens):
    token_map = {x: i + 3 for i, x in enumerate(tokens)}
    token_map['PAD'] = 0
    token_map['SOW'] = 1
    token_map['EOW'] = 2
    return token_map


def split_dataset(df):
    train, val, test = [], [], []
    train_split, val_split, test_split = .8, .1, .1
    assert train_split + val_split + test_split == 1, 'Split sizes need to sum to 1'

    for label in df.inflection_label.unique():
        df_label = df[df['inflection_label'] == label].copy()
        n_row = df_label.shape[0]

        train += [pop_df(df_label, int(n_row * train_split))]
        val += [pop_df(df_label, int(n_row * val_split))]
        test += [df_label.copy()]

    train, val, test = pd.concat(train), pd.concat(val), pd.concat(test)
    return train, val, test


def pop_df(df, size):
    df_pop = df.iloc[:int(size)].copy()
    df.drop(df.index[:int(size)], inplace=True)
    return df_pop


def build_word_info(df, lang, rare_mode):
    if lang not in constants.controls:
        return
    WordInfo.build(df, lang, rare_mode)


def process_data(train_df, val_df, test_df, token_map, labels, lang, rare_mode):
    process_data_split(train_df, token_map, lang, rare_mode, 'train')
    process_data_split(val_df, token_map, lang, rare_mode, 'val')
    process_data_split(test_df, token_map, lang, rare_mode, 'test')
    save_info(token_map, labels, lang, rare_mode)


def process_data_split(df, token_map, lang, rare_mode, mode):
    data = parse_data(df, token_map)
    save_data(data, lang, rare_mode, mode)


def parse_data(df, token_map):
    max_len = df['root'].map(lambda x: len(x)).max()
    data = np.zeros((df.shape[0], max_len + 4))

    for i, (index, x) in enumerate(df.iterrows()):
        instance = x['root']
        data[i, 0] = token_map['SOW']
        data[i, 1:len(instance) + 1] = [token_map[z] for z in instance]
        data[i, len(instance) + 1] = token_map['EOW']
        data[i, -2] = x.inflection_label
        data[i, -1] = x.item_id

    return data


def save_data(data, lang, rare_mode, mode):
    fname = 'data-%s-%s-%s.npy' % (lang, rare_mode, mode)
    fpath = ApplicationPaths.datasets(subfolder_path='preprocess', file_name=fname)
    with open(fpath, 'wb') as f:
        np.save(f, data)


def save_info(token_map, labels, lang, rare_mode):
    info = {
        'token_map': token_map,
        'labels': labels,
    }
    fname = 'data-%s-%s-info.pckl' % (lang, rare_mode)
    fpath = ApplicationPaths.datasets(subfolder_path='preprocess', file_name=fname)
    with open(fpath, 'wb') as f:
        pickle.dump(info, f)

    save_label_map(labels, lang, rare_mode)


def save_label_map(labels, lang, rare_mode):
    inflection_list = [[x, y] for x, y in labels.items()]
    inflection_map = pd.DataFrame(inflection_list, columns=['class name', 'class index'])
    fname = 'data-%s-%s-label_map.tsv' % (lang, rare_mode)
    fpath = ApplicationPaths.datasets(subfolder_path='preprocess', file_name=fname)
    inflection_map.to_csv(fpath, sep='\t', index=None)


def load_info(lang, rare_mode):
    fname = 'data-%s-%s-info.pckl' % (lang, rare_mode)
    fpath = ApplicationPaths.datasets(subfolder_path='preprocess', file_name=fname)
    with open(fpath, 'rb') as f:
        info = pickle.load(f)
    token_map = info['token_map']
    labels = info['labels']

    return token_map, labels


def parse(args):
    df, labels = get_data(args.lang, args.rare_mode)
    if args.lang not in ['ara']:
        df = filter_word2vec(df, args.lang, args.rare_mode)
    build_word_info(df, args.lang, args.rare_mode)
    print('Final number of words: %d' % (df.shape[0]))
    print_basic_stats(df)

    token_map = get_token_map(df)
    train, val, test = split_dataset(df)
    process_data(train, val, test, token_map, labels, args.lang, args.rare_mode)


if __name__ == '__main__':
    args = get_args()
    config_seed(seed=args.seed)
    parse(args)
