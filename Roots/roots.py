import numpy as np
from numpy import poly1d


class RootsNumberEstimator:
    @staticmethod
    def cartesian(f: poly1d, interval: tuple) -> int:
        signs = np.array([s for s in np.sign(f.coeffs) if s != 0])
        return RootsNumberEstimator._count_sign_changes(signs)

    @staticmethod
    def fourier(f: poly1d, interval: tuple) -> int:
        a, b = interval
        n = f.order
        a_signs = np.array(
            [s for s in np.sign([f.deriv(i)(a) for i in np.arange(n + 1)]) if s != 0])
        b_signs = np.array(
            [s for s in np.sign([f.deriv(i)(b) for i in np.arange(n + 1)]) if s != 0])

        return (RootsNumberEstimator._count_sign_changes(a_signs) -
                RootsNumberEstimator._count_sign_changes(b_signs))

    @staticmethod
    def sturm(polynomial: poly1d, interval: tuple) -> int:
        a, b = interval
        n = polynomial.order
        f_sequence = [polynomial, polynomial.deriv()]
        for i in np.arange(2, n + 1):
            f_sequence.append(-np.polydiv(f_sequence[i - 2], f_sequence[i - 1])[1])

        a_signs = np.array([s for s in np.sign([f(a) for f in f_sequence]) if s != 0])
        b_signs = np.array([s for s in np.sign([f(b) for f in f_sequence]) if s != 0])
        return (RootsNumberEstimator._count_sign_changes(a_signs) -
                RootsNumberEstimator._count_sign_changes(b_signs))

    @staticmethod
    def _count_sign_changes(sequence: np.ndarray) -> int:
        n = 0
        current_sign = sequence[0]
        for s in sequence:
            if s != current_sign:
                n += 1
                current_sign = s
        return n


class RootApproximation:
    @staticmethod
    def binomial_search(f: poly1d, interval: tuple, tol: float) -> float:
        a, b = interval
        c = (a + b) / 2

        if abs(a - b) < tol:
            return c

        if f(a) * f(c) < 0:
            return RootApproximation.binomial_search(f, (a, c), tol)
        elif f(b) * f(c) < 0:
            return RootApproximation.binomial_search(f, (c, b), tol)
        else:
            return c


class Solver:
    def __init__(self, **kwargs):
        self._n_estimator = kwargs['n_estimator']
        self._root_approximation = kwargs['root_approximation']
        self._tolerance = kwargs['tolerance']

        self._compute_polynomial_approximation()

    def _compute_polynomial_approximation(self):
        self._f = poly1d([1, -6, 11, -6])

    def solve(self) -> np.ndarray:
        return self._find_roots(self._f, self._localize())

    def _localize(self):
        # Choosing ring on which all roots lie
        a = abs(self._f.coeffs[-1]) / (abs(self._f.coeffs[-1]) + max(self._f.coeffs[:-1]))
        b = 1 + max(self._f.coeffs[1:]) / abs(self._f.coeffs[0])

        n_plus = self._n_estimator(self._f, (a, b))

        m = n_plus

        while True:
            interval = np.linspace(a, b, m)
            n = 0
            for i in np.arange(m - 1):
                if self._f(interval[i]) * self._f(interval[i + 1]) < 0:
                    n += 1
            if n == n_plus:
                return interval

            h = (b - a) / m
            if h < self._tolerance:
                return interval

            m *= 2

    def _find_roots(self, f: poly1d, subdivided_interval: np.ndarray):
        n_points = subdivided_interval.size
        roots = []
        for i in range(n_points - 1):
            if f(subdivided_interval[i]) * f(subdivided_interval[i + 1]) < 0:
                roots.append(self._root_approximation(f, (subdivided_interval[i],
                                                          subdivided_interval[i + 1]),
                                                      self._tolerance))
        return np.array(roots)


if __name__ == '__main__':
    solver = Solver(n_estimator=RootsNumberEstimator.sturm,
                    root_approximation=RootApproximation.binomial_search,
                    tolerance=1e-9)
    print(solver.solve())
