from typing import Tuple, Callable

import numpy as np
from numpy import poly1d


class PolynomialSolver:
    def __init__(self, n_estimator: Callable[[poly1d, Tuple[float, float]], int],
                 root_approximation: Callable[[poly1d, Tuple[float, float]], float]):
        self._n_estimator = n_estimator
        self._root_approximation = root_approximation

    def solve(self, f: poly1d, tolerance: float) -> np.ndarray:
        return self._find_roots(f, self._localize(f, tolerance), tolerance)

    def _localize(self, f: poly1d, tolerance: float):
        # Поиск границ кольца на котором лежат корни
        a = abs(f.coeffs[-1]) / (abs(f.coeffs[-1]) + max(f.coeffs[:-1]))
        b = 1 + max(f.coeffs[1:]) / abs(f.coeffs[0])
        print('Ring borders: [{0:.3f}, {1:.3f}]'.format(a, b))

        n_plus = self._n_estimator(f, (a, b))
        print('Estimated N roots: {}'.format(n_plus))

        m = n_plus

        while True:
            interval = np.linspace(a, b, m)
            n = 0
            for i in np.arange(m - 1):
                if f(interval[i]) * f(interval[i + 1]) < 0:
                    n += 1
            if n == n_plus:
                return interval

            h = (b - a) / m
            if h < tolerance:
                return interval

            m *= 2

    def _find_roots(self, f: poly1d, subdivided_interval: np.ndarray, tolerance: float):
        n_points = subdivided_interval.size
        roots = []
        for i in range(n_points - 1):
            if f(subdivided_interval[i]) * f(subdivided_interval[i + 1]) < 0:
                roots.append(self._root_approximation(f, (subdivided_interval[i],
                                                          subdivided_interval[i + 1]),
                                                      tolerance))
        return np.array(roots)


