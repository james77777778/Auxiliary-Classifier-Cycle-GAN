import random
import numpy as np
import torch


def dict2table(params):
    if not isinstance(params, dict):
        params = vars(params)
    text = '\n\n'
    text = '|  Attribute  |     Value    |\n'+'|'+'-'*13+'|'+'-'*14+'|'
    for key, value in params.items():
        text += '\n|{:13}|{:14}|'.format(str(key), str(value))
    return text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
