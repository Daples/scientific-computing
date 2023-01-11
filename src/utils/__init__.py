from _typing import sparray

import scipy.sparse as sp
import scipy.sparse.linalg as sla

import numpy as np


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

    Returns
    -------
    numpy.ndarray
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
    U = sp.triu(A, format="lil")
    L = sp.tril(A, format="lil")

    aux = 1 / omega * D + L
    M = omega / (2 - omega) @ aux * sla.inv(D) @ aux.T
    return M
