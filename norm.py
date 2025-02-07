import numpy as np


def norm(x):
    max = np.pi
    min = -np.pi
    y = ((x - min) / (max - min) - 0.5) * 2
    return y, min, max


def renorm(y, xmin, xmax):
    x = (y / 2 + 0.5) * (xmax - xmin) + xmin
    return x
