import numpy as np

import matplotlib.pyplot as plt


class DataTable:
    def __init__(self, x: np.ndarray, f: np.ndarray):
        assert x.ndim == 1 and f.ndim == 1, 'x and f have to be 1d arrays'
        assert x.size == f.size, 'x and f have to have equal size'
        self.x = x

        self.f = f

    def divided_diff(self, k: int) -> np.ndarray:
        n = self.x.size

        assert k < n, 'k have to be lesser than n'

        if k == 0:
            return self.f
        elif k == 1:
            b = np.zeros(n - 1)
            for i in range(0, n - 1):
                b[i] = (self.f[i + 1] - self.f[i]) / (self.x[i + 1] - self.x[i])
            return np.array(b)
        else:
            b = np.zeros(n - k)
            for i in range(n - k):
                for j in range(k + 1):
                    p = self.f[i + j]

                    for r in range(k + 1):
                        if r != j:
                            p /= self.x[i + j] - self.x[i + r]

                    b[i] += p

            return b

    def finite_diff(self, i: int) -> float:
        """
        Returns finite difference in point x_i.
        :param i: Num of point in range [0, n)
        :return: central finite difference
                 for midpoints, forward difference
                 for border points
        """
        n = self.x.size
        assert 0 <= i < n, 'i must be in range [0, n)'

        if i == 0:
            return (self.f[1] - self.f[0]) / (self.x[1] - self.x[0])
        elif i == n - 1:
            return (self.f[-1] - self.f[-2]) / (self.x[-1] - self.x[-2])
        else:
            return (self.f[i + 1] - self.f[i - 1]) / (self.x[i + 1] - self.x[i - 1])


def read_csv(file_name: str) -> DataTable:
    '''
    Read csv file to DataTable
    :param file_name:
    :return:
    '''
    file = open(file_name)
    x_name, y_name = file.readline().split(',')

    x_list = []
    f_list = []
    for s in file:
        x, f = s.split(',')
        x, f = float(x), float(f)
        x_list.append(x)
        f_list.append(f)

    return DataTable(np.array(x_list), np.array(f_list))


class Newton:
    def __init__(self, data: DataTable):
        self.n = data.x.size
        self.b = [data.divided_diff(i)[0] for i in range(self.n)]

        self.x = data.x
        self.f = data.f

    def __call__(self, x: float) -> float:
        p = 0
        for i in range(self.n):
            d = self.b[i]
            for j in range(0, i):
                d *= x - self.x[j]
            p += d

        return p

    def __str__(self):
        return str(self.b)


class CubicSpline:
    def __init__(self, data: DataTable):
        # Array for coeffs of cubic spline.
        self.data = data

    def __call__(self, x: float):
        assert self.data.x[0] <= x <= self.data.x[-1], 'x not in [x_0, x_n]'
        n = self.data.x.size
        for i in range(0, n - 1):
            if self.data.x[i] <= x <= self.data.x[i + 1]:
                x_i = self.data.x[i]
                x_i_1 = self.data.x[i + 1]

                f_i = self.data.f[i]
                f_i_1 = self.data.f[i + 1]

                df_i = data.finite_diff(i)
                df_i_1 = data.finite_diff(i + 1)

                sys_matrix = np.array([[x_i ** 3, x_i ** 2, x_i, 1],
                                       [x_i_1 ** 3, x_i_1 ** 2, x_i_1, 1],
                                       [3 * x_i ** 2, 2 * x_i, 1, 0],
                                       [3 * x_i_1 ** 2, 2 * x_i_1, 1, 0]])
                sys_rhs = np.array([f_i, f_i_1, df_i, df_i_1])

                a = np.linalg.solve(sys_matrix, sys_rhs)
                return a[0] * (x ** 3) + a[1] * (x ** 2) + a[2] * x + a[3]


if __name__ == '__main__':
    data = read_csv('table.csv')
    newton = Newton(data)
    spline = CubicSpline(data)

    x_data = np.linspace(data.x[0], data.x[-1])
    newton_data = np.array([newton(x) for x in x_data])
    spline_data = np.array([spline(x) for x in x_data])

    plt.plot(data.x, data.f, 'o')
    plt.plot(x_data, newton_data, label='newton')
    plt.plot(x_data, spline_data, label='spline')

    plt.legend()
    plt.show()
