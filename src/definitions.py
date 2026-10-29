from sys import stderr
from torch import nn
import numpy as np

builtin_print = print
def print(*values: object,
          sep: str | None = " ",
          end: str | None = "\n",
          file = stderr,
          flush = False,
          verbose = True):
    builtin_print(*values, sep=sep, end=end, file=file, flush=flush) if verbose else None

def subset(dictionary:dict, *keys):
    return {k: dictionary[k] for k in keys}

def seq_eagle_model(in_features, out_features, n_layers):
    reduction_factor = np.pow(out_features / in_features, 1 / n_layers)
    n_layers = n_layers - 1
    layers = []
    for j in range(n_layers - 1):
        l = nn.Linear(in_features, int(np.floor(in_features*reduction_factor)))
        layers.append(l)
        layers.append(nn.ReLU())
        in_features = int(np.floor(in_features*reduction_factor))
    l = nn.Linear(in_features, out_features)
    layers.append(l)
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def behav_eagle_model(in_features, out_features):
    return nn.Sequential(
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, out_features),
)


def split_if(criteria, *data):
    assert all([len(column) == len(data[0]) for column in data])
    select_row=lambda j: [l[j] for l in data]
    n = len(data)
    excluded = [[]]*n
    j = 0
    while j < len(data[0]):
        if criteria(select_row(j)):
            j += 1
        else:
            for k in range(n):
                excluded[k].append(data[k].pop(j)) # also: del X[i], y[i]
    return excluded


def select(*data, where=lambda x: True):
    assert all([len(column) == len(data[0]) for column in data])
    return list(zip(*[row for row in zip(*data) if where(row)]))
