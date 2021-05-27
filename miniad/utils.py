# encoding:utf-8
import numpy as np


def clip(x, eps):
    if isinstance(x, np.ndarray):
        x = np.where((x >= 0) & (x < eps), eps, x)
        x = np.where((x < 0) & (x > -eps), -eps, x)
    else:
        if 0 <= x < eps:
            x = eps
        elif -eps < x < 0:
            x = -eps

    return x
