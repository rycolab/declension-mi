import argparse
from . import utils

parser = argparse.ArgumentParser(description='Phoneme LM')

# Data
parser.add_argument('--languages', type=str, nargs='+',
                    default=['ger_n', 'cze'],
                    help='Languages used. (default: [ger_n, cze])')
parser.add_argument('--rare-modes', type=str, nargs='+', default=['drop'],
                    help='What to do with rare inflection classes. (default: [drop])')

# Model
parser.add_argument('--control-vars', default='all', choices=['all',
                                                              'none'],
                    help='Control for confounding variables. (default: all)')

# Model
parser.add_argument('--model', default='lstm', choices=['lstm',
                                                        'mlp-word2vec'],
                    help='Model used. (default: lstm)')
parser.add_argument('--context', default='none', choices=['none', 'word2vec'],
                    help='Context used for systematicity. (default: none)')

parser.add_argument('--opt', action='store_true', default=False,
                    help='Should use optimum parameters in training.')

# Others
parser.add_argument('--results-path', type=str, default='results',
                    help='Path where results should be stored.')
parser.add_argument('--checkpoint-path', type=str, default='checkpoints',
                    help='Path where checkpoints should be stored.')
parser.add_argument('--csv-folder', type=str, default=None,
                    help='Specific path where to save results.')
parser.add_argument('--seed', type=int, default=7,
                    help='Seed for random algorithms repeatability (default: 7)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, csv_folder='', orig_folder=True, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    csv_folder = csv_folder if csv_folder != 'normal' or not args.opt else 'opt'
    csv_folder = csv_folder if args.csv_folder is None else args.csv_folder

    if orig_folder:
        args.rfolder = '%s/%s/orig' % (args.results_path, csv_folder)  # Results folder
    else:
        args.rfolder = '%s/%s' % (args.results_path, csv_folder)  # Results folder

    args.cfolder = '%s/%s' % (args.checkpoint_path, csv_folder)  # Checkpoint folder
    utils.mkdir(args.rfolder)
    utils.mkdir(args.cfolder)
    utils.config_seed(args.seed)
    return args
