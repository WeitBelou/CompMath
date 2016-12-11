import numpy as np

from equation import Grid, Equation


class ButcherTable:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        self.s = a.shape[0]
        assert b.size == self.s, 'size of b differs'
        assert c.size == self.s, 'size of c differs'

        self.a = a
        self.b = b
        self.c = c


class RungeKuttaMethod:
    def __init__(self, table: ButcherTable):
        self.table = table

    def __call__(self, equation: Equation, grid: Grid):
        y = np.zeros(grid.n_points)
        y[0] = equation.y_0

        for n in range(1, grid.n_points):
            y[n] = y[n - 1]
            k = np.zeros(self.table.s)

            for i in range(self.table.s):
                x_ = grid[n - 1] + grid.h * self.table.c[i]
                y_ = y[n - 1]
                for j in range(i):
                    y_ += self.table.a[i, j] * grid.h * k[j]
                k[i] = equation.f(x_, y_)

            for i in range(self.table.s):
                y[n] += grid.h * k[i] * self.table.b[i]

        return y


def modified_euler() -> ButcherTable:
    a = np.array([[0.0, 0.0],
                  [0.5, 0.0]])

    b = np.array([0.0, 1.0])

    c = np.array([0.0, 0.5])

    return ButcherTable(a, b, c)


def euler_with_count() -> ButcherTable:
    a = np.array([[0.0, 0.0],
                  [1.0, 0.0]])
    b = np.array([0.5, 0.5])
    c = np.array([0.0, 1.0])
    return ButcherTable(a, b, c)


def heun_1() -> ButcherTable:
    a = np.array([[0.0, 0.0, 0.0],
                  [1 / 3, 0.0, 0.0],
                  [0.0, 2 / 3, 0.0]])
    b = np.array([1 / 4, 0.0, 3 / 4])
    c = np.array([0.0, 1 / 3, 2 / 3])
    return ButcherTable(a, b, c)


def heun_2() -> ButcherTable:
    a = np.array([[0.0, 0.0, 0.0],
                  [2 / 3, 0.0, 0.0],
                  [-1 / 3, 1.0, 0.0]])
    b = np.array([1 / 4, 1 / 2, 1 / 4])
    c = np.array([0.0, 2 / 3, 2 / 3])
    return ButcherTable(a, b, c)


def heun_3() -> ButcherTable:
    a = np.array([[0.0, 0.0, 0.0],
                  [1 / 2, 0.0, 0.0],
                  [-1.0, 2.0, 0.0]])
    b = np.array([1 / 6, 2 / 3, 1 / 6])
    c = np.array([0.0, 1 / 2, 1.0])
    return ButcherTable(a, b, c)


def runge_kutta_4() -> ButcherTable:
    a = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.5, 0.0, 0.0, 0.0],
                  [0.0, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    c = np.array([0.0, 0.5, 0.5, 1.0])
    return ButcherTable(a, b, c)
