import numpy as np
import random
import torch
import torch.nn.functional as F


def get_negative_mask(b_size):
    # remove similarity score of similar cascades
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/helpers.py
    negative_mask = np.ones((b_size, b_size * 2), dtype=bool)
    for i in range(b_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + b_size] = 0
    # torch.Tensor(negative_mask).bool()
    return ~torch.eye(b_size * 2, b_size * 2, dtype=bool)


def dot_sim_1(x, y):
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py
    return torch.matmul(x.unsqueeze(1), y.unsqueeze(2))


def dot_sim_2(x, y):
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py

    return torch.tensordot(x.unsqueeze(1), y.transpose(len(y.shape) - 1, 0).unsqueeze(0), dims=2)


def shuffle_two(x, y):
    couple = list(zip(x, y))
    random.shuffle(couple)
    return zip(*couple)


def divide_dataset(x, label_fractions=100):
    # only for 1%, 10%, and 100% label fractions
    return x[::100 // label_fractions]
