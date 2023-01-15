import inspect
from time import time
from typing import cast

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from _typing import sparray
from utils.result import IterativeResults

_tolerance: float = 1e-10


def fill_block_matrix(
    n: int, no_boundary_matrix: sparray, fill_subblocks: bool = True
) -> sparray:
    """It fills the matrix without elimination of boundary conditions.

    Parameters
    ----------
    n: int
        The number of elements in the discretization on each dimension.
    no_boundary_matrix: numpy.ndarray
        The matrix with elimination of boundary conditions.
    fill_subblocks: bool, optional
        If auxiliary blocks should be filled. Default: True

    Returns
    -------
    _typing.sparray
        The filled matrix with boundary conditions.
    """

    h = 1 / n
    dim = n**2 - 1
    matrix = sparray((dim, dim))

    diagonal_blocks = n - 1
    subdiagonal_blocks = n - 2

    block_width = n + 1
    block_idx = np.array([0, block_width])
    inner_block_width = n - 1
    inner_block_idx = np.array([0, inner_block_width])

    for i in range(diagonal_blocks):
        block = sparray((block_width, block_width))

        # Assign boundaries
        if fill_subblocks:
            block[[0, -1], [0, -1]] = h**2

        # Assign inner block
        idx_inner = inner_block_idx + inner_block_width * i
        block[1:-1, 1:-1] = no_boundary_matrix[
            idx_inner[0] : idx_inner[1], idx_inner[0] : idx_inner[1]
        ]

        # Assign block
        idx = block_idx + block_width * i
        matrix[idx[0] : idx[1], idx[0] : idx[1]] = block

        # Assign other blocks
        if i < subdiagonal_blocks and fill_subblocks:
            block = -sp.identity(block_width).tolil()
            block[[0, -1], [0, -1]] = 0
            shifted_idx = block_idx + block_width * (i + 1)

            # Lower
            matrix[shifted_idx[0] : shifted_idx[1], idx[0] : idx[1]] = block

            # Upper
            matrix[idx[0] : idx[1], shifted_idx[0] : shifted_idx[1]] = block

    return matrix


def get_M_SSOR(A: sparray, omega: float) -> sparray:
    """It returns the matrix M for BIM with SSOR.

    Parameters
    ----------
    A: _typing.sparray
        The symmetric matrix.

    Returns
    -------
    _typing.sparray
        The SSOR matrix.
    """

    D = sp.diags(A.diagonal(), format="lil")
    L = sp.tril(A, format="lil")

    aux = 1 / omega * D + L
    M = omega / (2 - omega) * aux @ sla.inv(D) @ aux.T
    return M


def solve_cg(
    A: sparray,
    f: np.ndarray,
    u_true: np.ndarray,
    u0: np.ndarray | None = None,
    M: sparray | np.ndarray | None = None,
) -> IterativeResults:
    """It uses sparse Conjugate Gradient to solve `Au = f`.

    Parameters
    ----------
    A: _typing.sparray
        The sparse matrix A.
    f: numpy.ndarray
        The right hand side of the equation.
    u0: numpy.ndarray | None, optional
        The initial guess vector. Default: None.
    M: _typing.sparray | numpy.ndarray | None, optional
        The precondition matrix.

    Returns
    -------
    utils.result.IterativeResults
        The results object.
    """

    residuals = []
    true_residuals = []
    t0 = time()
    iterations = 0
    exit_code = -1
    if u0 is None:
        u0 = np.empty(A.shape[0])
    results = IterativeResults(u0, residuals, true_residuals, t0, iterations, exit_code)

    def get_from_iteration(xk) -> None:
        # Update iteration
        results.iterations += 1

        # Store estimated residual
        frame = inspect.currentframe().f_back  # type: ignore
        results.residuals.append(frame.f_locals["resid"])  # type: ignore

        # Store true residual
        results.errors.append(cast(float, np.linalg.norm(xk - u_true)))

    u, info = sla.cg(A, f, tol=_tolerance, x0=u0, M=M, callback=get_from_iteration)
    results.time = time() - results.time
    results.solution = u
    results.exit_code = info

    return results
