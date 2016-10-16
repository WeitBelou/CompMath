from typing import Tuple, Callable

import numpy as np
from numpy import poly1d

import Roots.roots_number_estimators as rne
import Roots.root_approximators as ra


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


class Parameters:
    def __init__(self):
        self._parameters = dict()

    def parse_file(self, in_file_name: str):
        in_file = open(in_file_name)
        self._parameters = {s.strip().split('=')[0]: float(s.strip().split('=')[1])
                            for s in in_file.readlines() if s.strip()}

    def __getitem__(self, item):
        return self._parameters[item]


class ProblemSolver:
    def __init__(self, p: Parameters):
        alpha_0 = round((p['gamma_0'] + 1) / (p['gamma_0'] - 1))
        n = round(2 * p['gamma_3'] / (p['gamma_3'] - 1))
        mu = ((p['U_3'] - p['U_0']) * np.sqrt(((p['gamma_0'] - 1) * p['rho_0']) / (2 * p['P_0'])))

        # Здесь возможно была ошибка в методичке (стр. 18)
        nu = 2 * p['C_3'] / (p['gamma_3'] - 1) * np.sqrt(p['rho_0'] *
                                                         (p['gamma_0'] - 1) / (2 * p['P_0']))

        X = p['P_3'] / p['P_0']

        coeffs = np.zeros(2 * n + 1)
        coeffs[0] = 1 - (mu + nu) ** 2
        coeffs[1] = 2 * nu * (mu + nu)
        coeffs[2] = - (nu ** 2)
        coeffs[n] = - (2 + alpha_0 * ((mu + nu) ** 2)) * X
        coeffs[n + 1] = 2 * alpha_0 * nu * (mu + nu) * X
        coeffs[n + 2] = -alpha_0 * (nu ** 2) * X
        coeffs[2 * n] = X ** 2

        self._parameters = p
        self._f = poly1d(coeffs[::-1])

    def solve(self, solver: PolynomialSolver, tolerance: float) -> np.ndarray:
        print(self._f)
        z = solver.solve(self._f, tolerance)

        n = round(2 * self._parameters['gamma_3'] / (self._parameters['gamma_3'] - 1))
        p_1 = self._parameters['P_3'] * (z ** n)

        return p_1


if __name__ == '__main__':
    parameters = Parameters()
    parameters.parse_file('parameters.txt')

    solver = PolynomialSolver(rne.sturm,
                              ra.newton)

    problem_solver = ProblemSolver(parameters)
    answer = problem_solver.solve(solver, 1e-9)
    print(answer)
