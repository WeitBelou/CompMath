from typing import Callable

import numpy as np


class Equation:
    def __init__(self, f: Callable[[float, float], float], y_0: float, interval: tuple):
        self.f = f
        self.y_0 = y_0
        self.interval = interval


class Grid:
    def __init__(self, a: float, b: float, n_points: int):
        self.interval = (a, b)
        self.n_points = n_points
        self.h = (b - a) / (n_points - 1)

    def get_linspace(self, n: int) -> np.ndarray:
        return np.linspace(*self.interval, n)

    def __str__(self) -> str:
        return '(a, b) = ' + str(self.interval) + ', n = ' + str(self.n_points) + ', h = ' + str(self.h)

    def refined_grid(self):
        return Grid(*self.interval, self.n_points * 2 - 1)

    def __getitem__(self, i: int):
        return self.interval[0] + self.h * i
