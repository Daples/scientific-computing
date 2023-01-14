import numpy as np
import scipy.sparse as sp

from _typing import sparray


def one_dimensional_poisson(n: int) -> np.ndarray:
    """Gets the LHS Poisson matrix with Dirichlet boundary conditions.

    Parameters
    ----------
    n: int
        The number of elements in the discretization.

    Returns
    -------
    numpy.ndarray
        The 1D Poisson matrix of shape ((n + 1), (n + 1))
    """

    h = 1 / n
    poisson_matrix = np.zeros((n + 1, n + 1))
    inner_poisson_matrix = get_inner_poisson_1D(n)
    poisson_matrix[1:-1, 1:-1] = inner_poisson_matrix
    poisson_matrix[0, 0] = h**2
    poisson_matrix[-1, -1] = h**2

    return poisson_matrix


def get_inner_poisson_1D(n: int) -> sparray:
    """Returns the inner Poisson matrix in 1D.

    Parameters
    ----------
    n: int
        The number of elements in the discretization.

    Returns
    -------
    numpy.ndarray
        The inner matrix of shape ((n - 1), (n - 1)).
    """

    n -= 1
    diagonals = [
        2 * np.ones(n),
        -1 * np.ones(n - 1),
        -1 * np.ones(n - 1),
    ]
    inner_poisson_matrix = sparray(sp.diags(diagonals, [0, -1, 1]))  # type: ignore

    return inner_poisson_matrix
