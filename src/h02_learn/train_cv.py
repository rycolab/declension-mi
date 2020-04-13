import os
import sys
import random
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.parse import get_data as get_raw_data
from h02_learn.model import opt_params
from h02_learn.train import convert_to_loader, _run_language, write_csv, get_data
from utils import argparser
from utils import utils


full_results = [['lang', 'rare_mode', 'fold', 'avg_len', 'entropy', 'unconditional_entropy',
                 'test_loss', 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]


def get_full_data_loader(lang, rare_mode):
    train_loader, val_loader, test_loader, token_map, labels = \
        get_data(lang, rare_mode, args)
    full_data = merge_data_loaders([train_loader, val_loader, test_loader])
    return full_data, token_map, labels


def merge_data_loaders(data_loaders):
    n_items = sum([x.dataset.tensors[0].shape[0] for x in data_loaders])
    x_size = max([x.dataset.tensors[0].shape[1] for x in data_loaders])

    x, y, idx = np.zeros((n_items, x_size)), np.zeros((n_items)), np.zeros((n_items))
    start, end = 0, 0
    for loader in data_loaders:
        for batch_x, batch_y, batch_idx in loader:
            end += batch_x.size(0)
            x[start:end, :batch_x.size(1)] = batch_x.cpu()
            y[start:end] = batch_y.cpu()
            idx[start:end] = batch_idx.cpu()
            start = end

    return x, y, idx


def get_lang_df(lang, rare_mode):
    df, _ = get_raw_data(lang, rare_mode)
    return df


def get_ids(lang, rare_mode):
    df = get_lang_df(lang, rare_mode)
    instance_ids = sorted(list(df.item_id.unique()))
    random.shuffle(instance_ids)
    return instance_ids


def get_data_loaders_cv(fold, nfolds, full_data, token_map, labels, instance_ids, args, verbose=True):
    data_split = get_data_split_cv(fold, nfolds, instance_ids, full_data[2], args, verbose=verbose)

    train_loader = get_data_loader(full_data, data_split[0], token_map, 'train', args)
    val_loader = get_data_loader(full_data, data_split[1], token_map, 'val', args)
    test_loader = get_data_loader(full_data, data_split[2], token_map, 'test', args)

    return train_loader, val_loader, test_loader


def get_data_split_cv(fold, nfolds, instance_ids, valid_ids, args, verbose=True):
    ids = [x for x in instance_ids if x in valid_ids]
    return _get_data_split_cv(fold, nfolds, ids, verbose=verbose)


def _get_data_split_cv(fold, nfolds, instance_ids, verbose=True):
    part_size = int(len(instance_ids) / nfolds)
    test_fold = (fold + 1) % nfolds
    train_start_fold = 0 if test_fold > fold else (test_fold + 1)

    train = instance_ids[train_start_fold * part_size:fold * part_size]
    train += instance_ids[(fold + 2) * part_size:] if fold + 2 < nfolds else []
    val = instance_ids[fold * part_size:(fold + 1) * part_size] if fold + 1 < nfolds else \
        instance_ids[fold * part_size:]
    test = instance_ids[(test_fold) * part_size:(test_fold + 1) * part_size] if test_fold + 1 < nfolds else \
        instance_ids[(test_fold) * part_size:]

    if verbose:
        print('Train %d, Val %d, Test %d' % (len(train), len(val), len(test)))

    return (train, val, test)


def get_data_loader(full_data, ids, token_map, mode, args):
    data = split_data(full_data, ids, token_map, mode, args)
    return convert_to_loader(data, mode)


def split_data(full_data, ids, token_map, mode, args):
    data_partial = [(x, y, item_id) for x, y, item_id in zip(*full_data) if item_id in ids]
    max_len = max([len(x) for (x, _, _) in data_partial])

    data = np.zeros((len(data_partial), max_len + 2)).astype(int)
    data.fill(token_map['PAD'])
    for i, (x, y, item_id) in enumerate(data_partial):
        data[i, :len(x)] = x
        data[i, -2] = y
        data[i, -1] = item_id

    return data


def run_language_cv(lang, rare_mode, instance_ids, args, embedding_size=None,
                    hidden_size=256, word2vec_size=10, nlayers=1, dropout=0.2):
    global full_results
    full_data, token_map, labels = get_full_data_loader(lang, rare_mode)
    nfolds = 10
    avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc = 0, 0, 0, 0
    for fold in range(nfolds):
        print()
        print('Fold:', fold, end=' ')
        train_loader, val_loader, test_loader = get_data_loaders_cv(
            fold, nfolds, full_data, token_map, labels, instance_ids, args)
        avg_len, entropy, uncond_entropy, test_loss, test_acc, \
            best_epoch, val_loss, val_acc = _run_language(
                lang, rare_mode, train_loader, val_loader, test_loader, token_map, labels,
                args, embedding_size=embedding_size, hidden_size=hidden_size,
                word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout)

        full_results += [[lang, rare_mode, fold, avg_len, entropy, uncond_entropy,
                          test_loss, test_acc, val_loss, val_acc, best_epoch]]

        avg_test_loss += test_loss / nfolds
        avg_test_acc += test_acc / nfolds
        avg_val_loss += val_loss / nfolds
        avg_val_acc += val_acc / nfolds

        write_csv(full_results, '%s/%s__%s__full-results.csv' % (args.rfolder, args.model, args.context))

    return avg_len, entropy, uncond_entropy, avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc


def run_opt_language_cv(lang, rare_mode, instance_ids, args):
    embedding_size, hidden_size, word2vec_size, nlayers, dropout = opt_params.get_opt_params(lang, rare_mode, args)
    print('Optimum hyperparams emb-hs: %d, hs: %d, w2v: %d, nlayers: %d, drop: %.4f'
          % (embedding_size, hidden_size, word2vec_size, nlayers, dropout))

    return run_language_cv(lang, rare_mode, instance_ids, args,
                           embedding_size=embedding_size, hidden_size=hidden_size, word2vec_size=word2vec_size,
                           nlayers=nlayers, dropout=dropout)


def run_language_enveloper_cv(lang, rare_mode, instance_ids, args):
    if args.opt:
        return run_opt_language_cv(lang, rare_mode, instance_ids, args)
    else:
        return run_language_cv(lang, rare_mode, instance_ids, args)


def run_languages(args):
    results = [['lang', 'rare_mode', 'avg_len', 'entropy', 'unconditional_entropy',
                'test_loss', 'test_acc', 'val_loss', 'val_acc']]

    languages = utils.get_languages(args.languages, args.rare_modes)
    for i, (lang, rare_mode) in enumerate(languages):
        print()
        print('%d. Language %s (%s)' % (i, lang, rare_mode))

        instance_ids = get_ids(lang, rare_mode)
        avg_len, entropy, uncond_entropy, test_loss, test_acc, \
            val_loss, val_acc = run_language_enveloper_cv(lang, rare_mode, instance_ids, args)

        results += [[lang, rare_mode, avg_len, entropy, uncond_entropy, test_loss, test_acc, val_loss, val_acc]]

        write_csv(results, '%s/%s__%s__results.csv' % (args.rfolder, args.model, args.context))
    write_csv(results, '%s/%s__%s__results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    run_languages(args)
