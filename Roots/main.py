from Roots.parameters import Parameters
from Roots.problem_solver import ProblemSolver
from Roots.number_estimators import cartesian, fourier, sturm
from Roots.root_approximators import binomial_search, newton
from Roots.polynomial_solver import PolynomialSolver

if __name__ == '__main__':
    parameters = Parameters()
    parameters.parse_file('parameters.txt')

    solver = ProblemSolver(parameters)

    poly_solver = PolynomialSolver(sturm, binomial_search)

    answer = solver.solve(poly_solver, 1e-3)

    print(answer)