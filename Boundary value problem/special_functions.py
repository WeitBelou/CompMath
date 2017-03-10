from typing import Callable

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
