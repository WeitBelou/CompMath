from typing import Tuple

import numpy as np
from numpy import poly1d


def cartesian(f: poly1d, interval: Tuple[float, float]) -> int:
    signs = np.sign(f.coeffs)[np.nonzero(f.coeffs)]
    return _count_sign_changes(signs)


def fourier(f: poly1d, interval: Tuple[float, float]) -> int:
    a, b = interval
    n = f.order
    a_signs = np.sign([f.deriv(i)(a) for i in np.arange(n + 1)])
    a_signs = a_signs[np.nonzero(a_signs)]

    b_signs = np.sign([f.deriv(i)(b) for i in np.arange(n + 1)])
    b_signs = b_signs[np.nonzero(b_signs)]

    return (_count_sign_changes(a_signs) -
            _count_sign_changes(b_signs))


def sturm(polynomial: poly1d, interval: Tuple[float, float]) -> int:
    a, b = interval
    n = polynomial.order
    f_sequence = [polynomial, polynomial.deriv()]
    for i in np.arange(2, n + 1):
        f_sequence.append(-np.polydiv(f_sequence[i - 2], f_sequence[i - 1])[1])

    a_signs = np.sign([f(a) for f in f_sequence])
    a_signs = a_signs[np.nonzero(a_signs)]

    b_signs = np.sign([f(b) for f in f_sequence])
    b_signs = b_signs[np.nonzero(b_signs)]

    return (_count_sign_changes(a_signs) -
            _count_sign_changes(b_signs))


def _count_sign_changes(sequence: np.ndarray) -> int:
    n = 0
    current_sign = sequence[0]
    for s in sequence:
        if s != current_sign:
            n += 1
            current_sign = s
    return n
