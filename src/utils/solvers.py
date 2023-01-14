from time import time
from typing import cast

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from _typing import sparray
from utils.result import IterativeResults


def solve_ssor(
    A: np.ndarray,
    f: np.ndarray,
    u_true: np.ndarray,
    omega: float,
    tol: float = 1e-10,
) -> IterativeResults:
    """"""

    residuals = []
    errors = []
    t0 = time()
    iterations = 0
    exit_code = -1
    u0 = np.zeros(A.shape[0])
    results = IterativeResults(u0, residuals, errors, t0, iterations, exit_code)

    def is_finished(residual: np.ndarray) -> bool:
        norm = lambda x: np.linalg.norm(x, ord=2)
        val = norm(residual) / norm(f)
        print(val)
        return cast(bool, val <= tol)

    def get_from_iteration(xk: np.ndarray) -> None:
        # Update iteration
        results.iterations += 1

        # Store estimated residual
        results.residuals.append(np.linalg.norm(residual))  # type: ignore

        # Store true residual
        results.errors.append(cast(float, np.linalg.norm(xk - u_true)))  # type: ignore

    xold = u0
    residual = f - A @ xold
    U = np.triu(A, k=1)
    L = np.tril(A, k=-1)
    D = np.diag(np.diag(A))

    Lstar = D + omega * L
    Ustar = D + omega * U
    S1 = -omega * L + (1 - omega) * D
    S2 = -omega * U + (1 - omega) * D
    b = f

    c1 = np.linalg.solve(np.tril(Lstar), b)
    c2 = np.linalg.solve(np.triu(Ustar), D @ c1)
    c2 *= omega * (2 - omega)

    while not is_finished(residual):
        xhalf = np.linalg.solve(np.tril(Lstar), S2 @ xold)
        xnew = np.linalg.solve(np.triu(Ustar), S1 @ xhalf)
        xnew += c2
        residual = b - A @ xnew
        xold = xnew
        get_from_iteration(xnew)

    results.solution = xold
    return results


def solve_ssor_sparse(
    A: sparray,
    f: np.ndarray,
    u_true: np.ndarray,
    omega: float,
    tol: float = 1e-10,
    verbose: bool = False,
) -> IterativeResults:
    """"""

    residuals = []
    errors = []
    t0 = time()
    iterations = 0
    exit_code = -1
    n = A.shape[0]
    u0 = np.zeros(n)
    results = IterativeResults(u0, residuals, errors, t0, iterations, exit_code)

    def is_finished(residual: sparray) -> bool:
        norm = lambda x: np.linalg.norm(x, ord=2)
        val = norm(residual) / norm(f)
        if verbose:
            print(val)
        return cast(bool, val <= tol)

    def get_from_iteration(xk: sparray) -> None:
        # Update iteration
        results.iterations += 1

        # Store estimated residual
        results.residuals.append(np.linalg.norm(residual))  # type: ignore

        # Store true residual
        results.errors.append(cast(float, np.linalg.norm(xk - u_true)))  # type: ignore

    xold = u0
    residual = f - A @ xold
    U = sp.triu(A, k=1, format="lil")
    L = sp.tril(A, k=-1, format="lil")
    D = sp.diags(A.diagonal(), format="lil")

    Lstar = D + omega * L
    Ustar = D + omega * U
    S1 = -omega * L + (1 - omega) * D
    S2 = -omega * U + (1 - omega) * D
    b = f

    c1 = sla.spsolve(Lstar, b)
    c2 = sla.spsolve(Ustar, D @ c1)
    c2 *= omega * (2 - omega)

    while not is_finished(residual):
        xhalf = sla.spsolve(Lstar, S2 @ xold)
        xnew = sla.spsolve(Ustar, S1 @ xhalf)
        xnew += c2
        residual = b - A @ xnew
        xold = xnew
        get_from_iteration(xnew)

    results.solution = xold
    results.time = time() - results.time
    return results
