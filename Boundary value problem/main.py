from math import exp, sin
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import PiecewiseFunction, BoundaryConditions, Grid


class BoundaryValueProblem:
    """
    Класс представляющий задачу с граничными условиями и
    разрывными коэффициентами
    """

    def __init__(self, grid: Grid,
                 boundary_conditions: BoundaryConditions,
                 k: PiecewiseFunction, q: PiecewiseFunction,
                 f: PiecewiseFunction):
        self._x_0 = k.get_discontinuous_point()
        assert np.math.isclose(self._x_0, q.get_discontinuous_point()), 'Функции должны иметь одинаковую точку разрыва'
        assert np.math.isclose(self._x_0, f.get_discontinuous_point()), 'Функции должны иметь одинаковую точку разрыва'

        self._k = k
        self._q = q
        self._f = f

        self._boundary = boundary_conditions
        self._grid = grid

    def solve_analytical_model_problem(self) -> PiecewiseFunction:
        class Coeffs:
            def __init__(self, l, mu, k):
                self.l = l
                self.mu = mu
                self.k = k

        def _get_coeff_tuple(i: int) -> Coeffs:
            """
            Возвращает кортеж с аргументами (l_i, mu_i, k_i)
            :type i: int
            :return:
            """
            q = self._q.get_function(i)(self._x_0)
            k = self._k.get_function(i)(self._x_0)
            f = self._f.get_function(i)(self._x_0)

            l = np.math.sqrt(q / k)
            mu = f / q

            return Coeffs(l, mu, k)

        alpha = _get_coeff_tuple(0)
        beta = _get_coeff_tuple(1)

        a = self._grid[0]
        b = self._grid[-1]
        x_0 = self._x_0

        system_matrix = [[exp(alpha.l * x_0), exp(-alpha.l * x_0), -exp(beta.l * x_0), -exp(-beta.l * x_0)],
                         [0, 0, exp(beta.l * b), exp(-beta.l * b)],
                         [exp(alpha.l * a), exp(alpha.l * a), 0, 0],
                         [alpha.l * alpha.k * exp(alpha.l * x_0), -alpha.l * alpha.k * exp(-alpha.l * x_0),
                          -beta.l * beta.k * exp(beta.l * x_0), beta.l * beta.k * exp(-beta.l * x_0)]]
        system_rhs = [beta.mu - alpha.mu,
                      -beta.mu + self._boundary.get_condition(1),
                      -alpha.mu + self._boundary.get_condition(0),
                      0]

        c = np.linalg.solve(system_matrix, system_rhs)

        return PiecewiseFunction(self._x_0,
                                 lambda x: c[0] * exp(alpha.l * x) + c[1] * exp(-alpha.l * x) + alpha.mu,
                                 lambda x: c[2] * exp(beta.l * x) + c[3] * exp(-beta.l * x) + beta.mu,
                                 )

    def solve(self) -> Tuple[Grid, np.ndarray]:
        """
        Решает задачу численным методом
        :return: кортеж, где первый элемент - сетка, а второй значение функции в её узлах
        """
        h = self._grid.h()

        a = lambda i: self._k(self._grid[i] + h / 2)
        c = lambda i: self._k(self._grid[i] - h / 2)
        b = lambda i: -(a(i) + c(i) + self._q(self._grid[i]) * h ** 2)
        d = lambda i: -self._f(self._grid[i]) * h ** 2

        alpha, beta = self._forward_walk(a, b, c, d)

        u = self._initialize_solution(alpha, beta)

        self._backward_walk(alpha, beta, u)

        return self._grid, u

    def _backward_walk(self, alpha, beta, u):
        l_a = self._grid.find_pos(self._x_0)

        for i in range(l_a - 2, 0, -1):
            u[i] = alpha[i] * u[i + 1] + beta[i]

        n = len(self._grid)
        for i in range(l_a + 2, n - 1):
            u[i] = alpha[i] * u[i - 1] + beta[i]

        u[0] = self._boundary.get_condition(0)
        u[-1] = self._boundary.get_condition(1)

    def _initialize_solution(self, alpha, beta):
        n = len(self._grid)
        u = np.zeros(n)

        l_a = self._grid.find_pos(self._x_0)
        k_a = self._k.get_function(0)(self._grid[l_a])
        k_b = self._k.get_function(1)(self._grid[l_a])
        u[l_a] = (k_a * beta[l_a - 1] + k_b * beta[l_a + 2]) / (k_a * (1 - alpha[l_a - 1]) + k_b * (1 - alpha[l_a + 2]))
        u[l_a + 1] = u[l_a]
        u[l_a - 1] = alpha[l_a - 1] * u[l_a] + beta[l_a - 1]
        u[l_a + 2] = alpha[l_a + 2] * u[l_a + 1] + beta[l_a + 2]
        return u

    def _forward_walk(self, a, b, c, d):
        n = len(self._grid)

        alpha = np.zeros(n)
        beta = np.zeros(n)

        alpha[1] = - a(1) / b(1)
        beta[1] = (d(1) - c(1) * self._boundary.get_condition(0)) / b(1)

        l_a = self._grid.find_pos(self._x_0)
        for i in range(2, l_a):
            alpha[i] = - a(i) / (b(i) + c(i) * alpha[i - 1])
            beta[i] = (d(i) - c(i) * beta[i - 1]) / (b(i) + c(i) * alpha[i - 1])
        alpha[n - 2] = - c(n - 2) / b(n - 2)
        beta[n - 2] = (d(n - 2) - c(n - 2) * self._boundary.get_condition(1)) / b(n - 2)

        for i in range(n - 3, l_a + 1, -1):
            alpha[i] = - c(i) / (b(i) + a(i) * alpha[i + 1])
            beta[i] = (d(i) - a(i) * beta[i + 1]) / (b(i) + a(i) * alpha[i + 1])
        return alpha, beta


def solve_real_problem():
    grid = Grid(0, 1, 11)
    x_0 = 1 / np.math.sqrt(2)
    k_function = PiecewiseFunction(x_0,
                                   lambda x: exp(sin(x)),
                                   lambda x: exp(x))
    q_function = PiecewiseFunction(x_0,
                                   lambda x: 2,
                                   lambda x: 1)
    f_function = PiecewiseFunction(x_0,
                                   lambda x: exp(x),
                                   lambda x: exp(x))
    boundary = BoundaryConditions(0, 1)

    boundary_value_problem = BoundaryValueProblem(grid, boundary, k_function, q_function, f_function)

    y = boundary_value_problem.solve()[1]

    plt.grid(True)
    plt.scatter(grid, y)
    plt.show()


def solve_model_problem_twice():
    grid = Grid(0, 1, 11)
    x_0 = 1 / np.math.sqrt(2)
    k_function = PiecewiseFunction(x_0,
                                   lambda x: exp(sin(x_0)),
                                   lambda x: exp(sin(x_0)))
    q_function = PiecewiseFunction(x_0,
                                   lambda x: 1,
                                   lambda x: 1)
    f_function = PiecewiseFunction(x_0,
                                   lambda x: exp(x_0),
                                   lambda x: exp(x_0))
    boundary = BoundaryConditions(0, 1)

    boundary_value_problem = BoundaryValueProblem(grid, boundary, k_function, q_function, f_function)
    analytical_solution = boundary_value_problem.solve_analytical_model_problem()

    numeric_y = boundary_value_problem.solve()[1]
    analytical_y = [analytical_solution(x) for x in grid]

    plt.grid(True)
    plt.scatter(grid, numeric_y, c='r')
    plt.scatter(grid, analytical_y, c='b')
    plt.show()


if __name__ == '__main__':
    solve_real_problem()
    solve_model_problem_twice()
