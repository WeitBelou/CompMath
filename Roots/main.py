import numpy as np

import Roots.number_estimators as rne
import Roots.root_approximators as ra
import Roots.polynomial_solver as r

if __name__ == '__main__':
    solver = r.PolynomialSolver(rne.sturm,
                                ra.binomial_search)
    answer = solver.solve(np.poly1d([1, -1, 3, -4], True), 1e-6)

    print(answer)