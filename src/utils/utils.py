import numpy as np
import random
import pathlib
import torch

from . import constants


def mkdir(fdir):
    pathlib.Path(fdir).mkdir(parents=True, exist_ok=True)


def config_seed(seed=77):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def merge_dicts(x, y):
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def dict_to_vec(id_dict):
    max_idx = max(id_dict.keys())
    vec = np.ones(max_idx + 1) * -1
    vec[list(id_dict.keys())] = list(id_dict.values())
    return vec


def get_languages(languages, rare_modes):
    languages = \
        [(x, y) for x in languages if x in constants.languages_nouns for y in rare_modes]
    return languages


def test_permutations(values, num_tests):
    real_avg = np.mean(values)

    n = 0
    for _ in range(num_tests):
        permut = np.random.randint(0, 2, size=(len(values))) * 2 - 1
        random_avg = np.mean(values * permut)
        if random_avg >= real_avg:
            n += 1

    return n / num_tests
