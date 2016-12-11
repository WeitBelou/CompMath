import matplotlib.pyplot as plt

from runge_kutta_methods import *

def evaluate_error(y_2h: np.array, y_h: np.array, k: float) -> float:
    n = y_2h.size - 1
    diff = 0
    for i in range(n):
        diff += abs(y_2h[i] - y_h[2 * i])

    return abs(diff) / (n * (2 ** k - 1))


def solve(equation: Equation, grid: Grid, method: ButcherTable, eps: float) -> np.ndarray:
    runge_kutta = RungeKuttaMethod(method)

    y_2h = runge_kutta(equation, grid)
    y_h = runge_kutta(equation, grid.refined_grid())

    k = method.s
    error = evaluate_error(y_2h, y_h, k)
    while error > eps:
        y_2h = runge_kutta(equation, grid)
        y_h = runge_kutta(equation, grid.refined_grid())
        error = evaluate_error(y_2h, y_h, k)

    print('Success', grid.refined_grid(), 'eps={}'.format(error))
    return y_h


def main():
    equation = Equation(lambda x, y: x / (y - x ** 2), 1.5, (1, 2))
    grid = Grid(*equation.interval, 11)
    eps = 1e-4

    y_array = solve(equation, grid, heun_1(), eps)
    x_array = grid.get_linspace(y_array.size)
    plt.plot(x_array, y_array, 'bo-')

    plt.show()


if __name__ == '__main__':
    main()
