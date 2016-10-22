import numpy as np
from numpy import poly1d

from Roots.polynomial_solver import PolynomialSolver
from Roots.parameters import Parameters


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

        self._p = p
        self._f = poly1d(coeffs[::-1])

    def solve(self, solver: PolynomialSolver, tolerance: float) -> np.ndarray:
        print(self._f)
        z = solver.solve(self._f, tolerance)
        u_1 = self._p['U_3'] + 2 * self._p['C_3'] * (1 - z) / (self._p['gamma_3'] - 1)

        n = round(2 * self._p['gamma_3'] / (self._p['gamma_3'] - 1))
        p_1 = self._p['P_3'] * (z ** n)

        return self._p['U_0'] - (self._p['P_0'] - p_1) / (self._p['rho_0'] * (u_1 - self._p['U_0']))