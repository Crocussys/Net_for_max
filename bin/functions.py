from bin.Matrix import Vector
import math


def derivative(function, var, val, step=0.001, *args, **kwargs):
    if isinstance(val, Vector):
        kwargs.update({var: val.copy() + step})
        f_plus = function(*args, **kwargs)
        kwargs.update({var: val.copy() - step})
        f_minus = function(*args, **kwargs)
    else:
        kwargs.update({var: val + step})
        f_plus = function(*args, **kwargs)
        kwargs.update({var: val - step})
        f_minus = function(*args, **kwargs)
    return (f_plus - f_minus) / (2 * step)


def sigmoid(x):
    if x < -709:
        return 0
    else:
        return 1 / (1 + math.exp(-x))


def relu(x):
    if x > 1e+50:
        return 1e+50
    if x >= 0:
        return x
    else:
        return 0


def loss(x, y):
    return sum((x - y) ** 2) / len(x)
