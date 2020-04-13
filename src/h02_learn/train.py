import os
import sys
import numpy as np
import math
import csv
from tqdm import tqdm
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.parse import load_info
from h01_data.word_info import WordInfo
from h02_learn.model import opt_params
from h02_learn.model import LstmLM, Word2VecMLP
from utils import argparser
from utils import utils
from utils.paths import ApplicationPaths
from utils import constants

results_per_word = [['lang', 'rare_mode', 'phoneme_id', 'phoneme', 'phoneme_len',
                     'target', 'predict', 'probabilities', 'phoneme_loss']]


def write_csv(results, filename):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def get_data(lang, rare_mode, args):
    token_map, labels = load_info(lang, rare_mode)
    train_loader, val_loader, test_loader = get_data_loaders(lang, rare_mode, args)
    return train_loader, val_loader, test_loader, token_map, labels


def get_data_loaders(lang, rare_mode, args):
    train_loader = get_data_loader(lang, rare_mode, 'train', args)
    val_loader = get_data_loader(lang, rare_mode, 'val', args)
    test_loader = get_data_loader(lang, rare_mode, 'test', args)

    return train_loader, val_loader, test_loader


def get_data_loader(lang, rare_mode, mode, args):
    data = read_data(lang, rare_mode, mode, args)
    return convert_to_loader(data, mode)


def read_data(lang, rare_mode, mode, args):
    fname = 'data-%s-%s-%s.npy' % (lang, rare_mode, mode)
    fpath = ApplicationPaths.datasets(subfolder_path='preprocess', file_name=fname)
    with open(fpath, 'rb') as f:
        data = np.load(f)

    return data


def convert_to_loader(data, mode, batch_size=64):
    x = torch.from_numpy(data[:, :-2]).long().to(device=constants.device)
    y = torch.from_numpy(data[:, -2]).long().to(device=constants.device)
    idx = torch.from_numpy(data[:, -1]).long().to(device=constants.device)

    shuffle = True if mode == 'train' else False

    dataset = TensorDataset(x, y, idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(train_loader, model, loss, optimizer):
    model.train()
    total_loss = 0.0
    for batches, (batch_x, batch_y, batch_idx) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(batch_x, batch_idx)
        l = loss(y_hat, batch_y)
        l.backward()
        optimizer.step()

        total_loss += l.item() / math.log(2)
    return total_loss / (batches + 1)


def eval(data_loader, model, loss):
    model.eval()
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    for _, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
        y_hat = model(batch_x, batch_idx)
        l = loss(y_hat, batch_y) / math.log(2)

        batch_size = batch_y.size(0)
        val_loss += l.item() * batch_size
        val_acc += (y_hat.argmax(-1) == batch_y).float().sum().item()
        total_sent += batch_size

    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    return val_loss, val_acc


def run_model(model, batch_x, batch_idx):
    return model(batch_x, batch_idx)


def eval_per_word(lang, rare_mode, data_loader, model, token_map, args, model_func=run_model):
    global results_per_word
    model.eval()

    token_map_inv = {x: k for k, x in token_map.items()}
    ignored_tokens = [token_map['PAD'], token_map['SOW'], token_map['EOW']]
    loss = nn.CrossEntropyLoss(reduction='none').to(device=constants.device)
    val_loss, val_acc, total_sent = 0.0, 0.0, 0

    for batches, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
        y_hat = model_func(model, batch_x, batch_idx)
        l = loss(y_hat, batch_y).detach() / math.log(2)

        batch_size = batch_y.size(0)
        val_loss += l.sum().item()
        val_acc += (y_hat.argmax(-1) == batch_y).float().sum().item()
        total_sent += batch_size

        words = batch_x.detach()
        words_len = (batch_x != 0).sum(-1) - 2
        probs = torch.softmax(y_hat, -1).gather(-1, batch_y.unsqueeze(1))
        for i, w in enumerate(words):
            # _w = [token_map_inv[x] for x in w.tolist() if x not in ignored_tokens]
            _w = idx_to_word(w, token_map_inv, ignored_tokens)
            idx = batch_idx[i].item()
            results_per_word += [[lang, rare_mode, idx, _w, words_len[i].item(), batch_y[i].item(),
                                  y_hat[i].argmax(-1).item(), probs[i].item(), l[i].item()]]

    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    write_csv(results_per_word, '%s/%s__%s__results-per-word.csv' % (args.rfolder, args.model, args.context))

    return val_loss, val_acc, results_per_word


def word_to_tensors(word, token_map):
    w = word_to_idx(word, token_map)

    x = torch.from_numpy(w[:, :-1]).long().to(device=constants.device)
    y = torch.from_numpy(w[:, 1:]).long().to(device=constants.device)
    return x, y


def word_to_idx(word, token_map):
    w = [[token_map['SOW']] + [token_map[x] for x in word] + [token_map['EOW']]]
    return np.array(w)


def idx_to_word(word, token_map_inv, ignored_tokens):
    _w = [token_map_inv[x] for x in word.tolist() if x not in ignored_tokens]
    return ' '.join(_w)


def train(train_loader, val_loader, test_loader, model, loss, optimizer, wait_epochs=10):
    epoch, best_epoch, best_loss, best_acc = 0, 0, float('inf'), 0.0

    pbar = tqdm(total=wait_epochs)
    while True:
        epoch += 1

        total_loss = train_epoch(train_loader, model, loss, optimizer)
        val_loss, val_acc = eval(val_loader, model, loss)

        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc
            model.set_best()

        pbar.total = best_epoch + wait_epochs
        pbar.update(1)
        pbar.set_description(
            '%d/%d: train_loss %.4f  val_loss: %.4f  val_acc: %.4f  best_loss: %.4f  best_acc: %.4f' %
            (epoch, best_epoch, total_loss, val_loss, val_acc, best_loss, best_acc))

        if epoch - best_epoch >= wait_epochs:
            break

    pbar.close()
    model.recover_best()

    return best_epoch, best_loss, best_acc


def _get_avg_len(data_loader):
    total_phon, total_sent = 0.0, 0.0
    for batches, (batch_x, _, _) in enumerate(data_loader):
        total_phon += (batch_x != 0).sum().item()
        total_sent += batch_x.size(0)

    avg_len = (total_phon * 1.0 / total_sent) - 2  # Remove SOW and EOW tag in every sentence

    return avg_len, total_sent


def get_avg_len(data_loaders):
    total_len, total_nsent = 0, 0
    for data_loader in data_loaders:
        length, nsentences = _get_avg_len(data_loader)
        total_len += (length * nsentences)
        total_nsent += nsentences

    return total_len * 1.0 / total_nsent


def get_unconditional_frequencies(data_loader):
    frequencies = {}
    for _, batch_y, _ in data_loader:
        for label in batch_y.unique().cpu().numpy():
            frequencies[label] = frequencies.get(label, 0) + (batch_y == label).sum().item()

    return frequencies


def get_unconditional_entropy(data_loaders):
    freq = {}
    for data_loader in data_loaders:
        new_freq = get_unconditional_frequencies(data_loader)
        freq = utils.merge_dicts(freq, new_freq)
    total_freq = sum(freq.values())

    probs = [x / total_freq for x in freq.values()]
    entropy = -sum([x * math.log(x, 2) for x in probs])
    return entropy, probs


def get_control_types(controls):
    if controls is None or not controls:
        return []

    future_controls = get_control_types(controls[1:])
    if future_controls:
        return [[x] + y for x in controls[0]['control_map'].values() for y in future_controls]
    else:
        return [[x] for x in controls[0]['control_map'].values()]


def get_conditional_frequencies(data_loader, control_vecs, control_classes):
    frequencies = {}
    for _, batch_y, batch_idx in data_loader:
        batch_controls2 = [(vec[batch_idx.cpu()] == control_class)
                           for vec, control_class in zip(control_vecs, control_classes)]
        batch_controls = np.array(batch_controls2).all(axis=0)

        batch_class = batch_y.cpu().numpy()[batch_controls]
        for label in np.unique(batch_class):
            frequencies[label] = frequencies.get(label, 0) + (batch_class == label).sum().item()

    return frequencies


def _get_conditional_entropy(data_loaders, control_vecs, control_classes):
    freq = {}
    for data_loader in data_loaders:
        new_freq = get_conditional_frequencies(data_loader, control_vecs, control_classes)
        freq = utils.merge_dicts(freq, new_freq)
    total_freq = sum(freq.values())

    probs = [x / total_freq for x in freq.values()]
    entropy = -sum([x * math.log(x, 2) for x in probs])
    return entropy, total_freq


def get_conditional_entropy(data_loaders, lang, rare_mode, args):
    controls_full = WordInfo.load(lang, rare_mode)
    control_classes_full = get_control_types([x for x in controls_full.values()])
    control_vecs = [utils.dict_to_vec(control['controls']) for control in controls_full.values()]

    entropies = []
    for control_classes in control_classes_full:
        entropy, freq = _get_conditional_entropy(data_loaders, control_vecs, control_classes)
        entropies += [(entropy, freq)]

    total_freq = sum([freq for _, freq in entropies])
    entropy = sum([h * freq / total_freq for h, freq in entropies])

    return entropy


def get_base_entropy(data_loaders, lang, rare_mode, args):
    control_vars = get_control_vars(lang, args)
    uncond_entropy, probs = get_unconditional_entropy(data_loaders)
    if control_vars is None:
        entropy = uncond_entropy
    else:
        entropy = get_conditional_entropy(data_loaders, lang, rare_mode, args)
    return entropy, uncond_entropy, probs


def get_control_vars(lang, args):
    if args.control_vars == 'all' and lang in constants.controls:
        return constants.controls[lang]
    elif args.control_vars != 'none' and args.control_vars != 'all':
        return [args.control_vars]

    return None


def init_model(model_name, lang, rare_mode, context, hidden_size, word2vec_size, token_map, labels,
               embedding_size, nlayers, dropout, args):
    vocab_size = len(token_map)
    n_classes = len(labels)
    control_vars = get_control_vars(lang, args)
    if model_name == 'lstm':
        model = LstmLM(
            vocab_size, n_classes, hidden_size, lang, rare_mode, embedding_size=embedding_size,
            word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout, context=context,
            controls=control_vars).to(device=constants.device)
    elif model_name == 'mlp-word2vec':
        model = Word2VecMLP(
            vocab_size, n_classes, hidden_size, lang, rare_mode, word2vec_size=word2vec_size,
            nlayers=nlayers, dropout=dropout, context=context, controls=control_vars) \
            .to(device=constants.device)
    else:
        raise ValueError("Model not implemented: %s" % model_name)

    return model


def get_model_entropy(
        lang, rare_mode, train_loader, val_loader, test_loader, token_map, labels, embedding_size, hidden_size, word2vec_size,
        nlayers, dropout, args, wait_epochs=10, per_word=True):
    model = init_model(
        args.model, lang, rare_mode, args.context, hidden_size, word2vec_size, token_map, labels,
        embedding_size, nlayers, dropout, args)

    loss = nn.CrossEntropyLoss().to(device=constants.device)
    optimizer = optim.Adam(model.parameters())

    best_epoch, val_loss, val_acc = train(train_loader, val_loader, test_loader, model,
                                          loss, optimizer, wait_epochs=wait_epochs)

    if per_word:
        test_loss, test_acc, _ = eval_per_word(lang, rare_mode, test_loader, model, token_map, args)
    else:
        test_loss, test_acc = eval(test_loader, model, loss)
    model.save(args.cfolder, args.context, '%s-%s' % (lang, rare_mode))

    return test_loss, test_acc, best_epoch, val_loss, val_acc


def _run_language(
        lang, rare_mode, train_loader, val_loader, test_loader, token_map, labels, args, embedding_size=None,
        hidden_size=256, word2vec_size=10, nlayers=1, dropout=0.2, per_word=True):
    avg_len = get_avg_len([train_loader, val_loader, test_loader])
    entropy, uncond_entropy, probs = get_base_entropy([train_loader, val_loader, test_loader], lang, rare_mode, args)
    print('Language %s (%s) Avg len: %.4f Entropy: %.2f Uncond Entropy: %.2f' %
          (lang, rare_mode, avg_len, entropy, uncond_entropy))
    print('Classes probabilities:', probs)

    test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
        lang, rare_mode, train_loader, val_loader, test_loader, token_map, labels, embedding_size, hidden_size,
        word2vec_size, nlayers, dropout, args, per_word=per_word)
    print('Test loss: %.4f  acc: %.4f    Avg len: %.4f' % (test_loss, test_acc, avg_len))

    return avg_len, entropy, uncond_entropy, test_loss, test_acc, best_epoch, val_loss, val_acc


def run_language(lang, rare_mode, args, embedding_size=None, hidden_size=256, word2vec_size=10, nlayers=1, dropout=0.2):
    train_loader, val_loader, test_loader, token_map, labels = get_data(lang, rare_mode, args=args)

    return _run_language(lang, rare_mode, train_loader, val_loader, test_loader, token_map, labels,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout)


def run_opt_language(lang, rare_mode, args):
    train_loader, val_loader, test_loader, token_map, labels = get_data(lang, rare_mode, args=args)
    embedding_size, hidden_size, word2vec_size, nlayers, dropout = opt_params.get_opt_params(lang, rare_mode, args)
    print('Optimum hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    return _run_language(lang, rare_mode, train_loader, val_loader, test_loader, token_map, labels,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout)


def run_language_enveloper(lang, rare_mode, args):
    if args.opt:
        return run_opt_language(lang, rare_mode, args)
    else:
        return run_language(lang, rare_mode, args)


def run_languages(args):
    results = [['lang', 'rare_mode', 'avg_len', 'entropy', 'unconditional_entropy',
                'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]

    languages = utils.get_languages(args.languages, args.rare_modes)
    for i, (lang, rare_mode) in enumerate(languages):
        print()
        print(i, end=' ')

        avg_len, entropy, uncond_entropy, test_loss, test_acc, \
            best_epoch, val_loss, val_acc = run_language_enveloper(lang, rare_mode, args)

        results += [[lang, rare_mode, avg_len, entropy, uncond_entropy,
                     test_loss, test_acc, best_epoch, val_loss, val_acc]]

        write_csv(results, '%s/%s__%s__results.csv' % (args.rfolder, args.model, args.context))
    write_csv(results, '%s/%s__%s__results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='normal')
    run_languages(args)
