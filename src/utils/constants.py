import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

controls = {
    'ger_n': ['gender'],
    'cze': ['gender'],
}

languages_nouns = ['ger_n', 'cze']
rare_modes = ['drop', 'group']
