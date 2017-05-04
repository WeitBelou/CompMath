from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def exact_solution(x: np.ndarray, t: np.ndarray):
    return np.cos(x - 2 * t)


def main():
    fig = plt.figure()

    plot_exact_solution(fig)
    plot_approx_solution(fig)

    plt.show()


def plot_exact_solution(fig):
    ax = fig.add_subplot(121, projection='3d')
    X = np.linspace(0, 1, 11)
    T = np.linspace(0, 1, 11)
    X, T = np.meshgrid(X, T)
    Z = exact_solution(X, T)
    surf = ax.plot_surface(X, T, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


def plot_approx_solution(fig):
    ax = fig.add_subplot(122, projection='3d')

    x = np.linspace(0, 1, 11)
    t = np.linspace(0, 1, 11)

    X, T = np.meshgrid(x, t)

    U = np.zeros(X.shape)

    # Граничные условия
    U[:, 0] = np.cos(x)

    # Начальные условия
    U[0, :] = np.cos(2 * t)

    h = 0.1
    tau = 0.1
    # Расширенные граничные условия
    U[1, :] = U[0] + h * np.sin(2 * t) - (h ** 2) * np.cos(2 * t) / 2 - (h ** 3) * np.sin(t) / 6
    U[2, :] = U[0] + 2 * h * np.sin(2 * t) - 2 * (h ** 2) * np.cos(2 * t) - 4 * (h ** 3) * np.sin(2 * t) / 3

    # В остальных точках
    L = 11
    N = 11

    for n in np.arange(1, N - 1):
        for l in np.arange(3, L):
            U[l, n + 1] = U[l, n] + tau / (3 * h) * (
                2 * U[l - 3, n] - 9 * U[l - 2, n] + 18 * U[l - 1, n] - 11 * U[l, n]) + 2 * (tau ** 2) / (h ** 2) * (
                - U[l - 3, n] + 4 * U[l - 2, n] - 3 * U[l - 1, n] + 2 * U[l, n]) - 4 * (tau ** 3) / (3 * h ** 3) * (
                - U[l - 3, n] + 3 * U[l - 2, n] - 3 * U[l - 1, n] + U[l, n])

    surf = ax.plot_surface(X, T, U, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


if __name__ == '__main__':
    main()
