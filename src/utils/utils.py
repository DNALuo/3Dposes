# util.py
# TODO:  This file provides some utilities for the model to
#    train, like adjusting learning rate and some others.
from numpy.random import randn


def set_sequence_length(min_seq_len_index, max_seq_len_index, epoch):
    if epoch <= max_seq_len_index:
        seq_len = 2 ** (min_seq_len_index + epoch)
    else:
        seq_len = 2 ** max_seq_len_index
    return seq_len


def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    # operator '//': divide with integral result (discard remainder)
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def Rnd(x):
    return max(-2 * x, min(2 * x, randn() * x))


def Flip(img):
    return img[:, :, ::-1].copy()


def ShuffleLR(x):
    shuffleRef = [[1, 2], [3, 4], [5, 6],
                  [7, 8], [9, 10], [11, 12]]
    for e in shuffleRef:
        x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
    return x
