from time import time
from typing import cast

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from _typing import sparray
from utils.result import IterativeResults


def solve_ssor_sparse(
    A: sparray,
    f: np.ndarray,
    u_true: np.ndarray,
    omega: float,
    tol: float = 1e-10,
    verbose: bool = False,
) -> IterativeResults:
    """It uses SSOR to solve `Au = f`.

    Parameters
    ----------
    A: _typing.sparray
        The left-hand matrix.
    f: np.ndarray
        The right-hand side.
    u_true: numpy.ndarray
        The exact solution (only for storing errors).
    omega: float
        The damping parameter.
    tol: float, optional
        The convergence tolerance. Default: 1e-10
    verbose: bool, optional
        If the residual should be printed after every iteration (debugging mostly).
        Default: False

    Returns
    -------
    utils.result.IterativeResults
        The results object.
    """

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
