from numpy import poly1d
from typing import Callable, Tuple


def binomial_search(f: Callable[[float], float],
                    interval: Tuple[float, float],
                    tol: float) -> float:
    a, b = interval

    c = (a + b) / 2
    while abs(a - b) > tol:
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c


def newton(f: poly1d, interval: Tuple[float, float], tol: float) -> float:
    a, b = interval
    x_0 = (a + b) / 2

    assert f(x_0) * f.deriv(2)(x_0) > 0, 'Не выполнены достаточные условия' \
                                         'сходимости метода Ньютона'

    x_1 = x_0 - f(x_0) / f.deriv(1)(x_0)

    while abs(x_1 - x_0) > tol:
        x_0 = x_1
        x_1 = x_0 - f(x_0) / f.deriv(1)(x_0)

    return x_1