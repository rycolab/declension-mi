import pandas as pd


def _get_opt_params(fname, lang, rare_mode, delimiter='\t'):
    results = pd.read_csv(fname, delimiter=delimiter)
    instance = results[results['lang'] == lang]
    if rare_mode != '':
        instance = instance[instance['rare_mode'] == rare_mode]

    embedding_size = int(instance['embedding_size'].item())
    hidden_size = int(instance['hidden_size'].item())
    word2vec_size = int(instance['word2vec_size'].item())
    nlayers = int(instance['nlayers'].item())
    dropout = instance['dropout'].item()

    return embedding_size, hidden_size, word2vec_size, nlayers, dropout


def get_opt_params(lang, rare_mode, args):
    context = args.context if 'shuffle' not in args.context else args.context[:-8]
    fname = '%s/bayes-opt/orig/%s__%s__opt-results.csv' \
        % (args.results_path, args.model, context)
    return _get_opt_params(fname, lang, rare_mode, delimiter=',')
