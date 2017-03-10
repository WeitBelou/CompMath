from typing import Callable

import numpy as np

USUAL_FUNCTION = Callable[[float], float]


class BoundaryConditions:
    def __init__(self, left: float, right: float):
        self._conditions = (left, right)

    def get_condition(self, id: int) -> float:
        try:
            return self._conditions[id]
        except IndexError:
            raise ValueError("Границы с номером {0} не существует, попробуйте номера 0 или 1".format(id))


class PiecewiseFunction:
    """
    Кусочно-непрерывная функция с одной точкой
    разрыва
    """

    def __init__(self, x_0: float,
                 left_fun: USUAL_FUNCTION,
                 right_fun: USUAL_FUNCTION):
        """
        Принимает точку разрыва и две непрерывные функции
        :type x_0: float
        :type left_fun: Callable[[float], float]
        :type right_fun: Callable[[float], float]
        """
        self._x_0 = x_0
        self._fun = (left_fun, right_fun)

    def __call__(self, x: float) -> float:
        return self._choose_function(x)(x)

    def get_discontinuous_point(self) -> float:
        return self._x_0

    def get_function(self, id: int) -> USUAL_FUNCTION:
        try:
            return self._fun[id]
        except IndexError:
            raise ValueError("Функции с номером {0} не существует, попробуйте номера 0 или 1".format(id))

    def _choose_function(self, x: float) -> USUAL_FUNCTION:
        """
        Возвращает функцию, действующую в данной точке
        :param x:
        :return: левую функцию, если x < x_0, иначе правую.
        """
        if x < self._x_0:
            return self._fun[0]
        else:
            return self._fun[1]


class Grid:
    """
    Простая обёртка над одномерным массивом
    """

    def __init__(self, left, right, n_points):
        self._grid = np.linspace(left, right, n_points, True)
        self._h = (right - left) / (n_points - 1)

    def h(self):
        return self._h

    def find_pos(self, x: float) -> int:
        """
        Возвращает положение перед разрывом
        :param x:
        :return:
        """
        i = 0
        while self._grid[i] < x:
            i += 1
        return i - 1

    def __getitem__(self, i: int) -> float:
        return self._grid[i]

    def __len__(self) -> int:
        return self._grid.size

    def __repr__(self) -> str:
        """
        Отладочное представление сетки
        :return: строковое представление
        """
        return str(self._grid)