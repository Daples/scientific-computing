import numpy as np
import scipy.sparse as sp

from _typing import sparray
from poisson.one_dimensional import get_inner_poisson_1D
from utils import fill_block_matrix


def two_dimensional_poisson(n: int) -> sparray:
    """It returns the Poisson matrix without elimination of boundary conditions.

    Parameters
    ----------
    n: int
        The number of elements to discretize each dimension.

    Returns
    -------
    _typing.sparray
        The matrix in sparse format of shape ((n + 1)**2, (n + 1)**2).
    """

    h = 1 / n
    diagonal_identity = sp.identity(n + 1) * h**2
    rectangular_zeros = sp.lil_matrix((n + 1, n**2 - 1))
    square_zeros = sp.lil_matrix((n + 1, n + 1))
    inner_poisson_matrix = get_inner_poisson_2D(n)
    poisson_matrix = sp.bmat(
        [
            [diagonal_identity, rectangular_zeros, square_zeros],
            [rectangular_zeros.T, inner_poisson_matrix, rectangular_zeros.T],
            [square_zeros, rectangular_zeros, diagonal_identity],
        ]
    ).tolil()

    return poisson_matrix / h**2


def get_inner_poisson_2D(n: int) -> sparray:
    """Returns the inner Poisson matrix in 2D.

    Parameters
    ----------
    n: int
        The number of elements in the discretization.

    Returns
    -------
    _typing.sparray
        The inner matrix of shape ((n**2 - 1), (n**2 - 1)).
    """

    no_boundary_matrix = _inner_no_boundary(n)
    matrix = fill_block_matrix(n, no_boundary_matrix)

    return matrix


def _inner_no_boundary(n: int) -> sparray:
    """Returns a 2D Poisson matrix with elimination of boundary conditions.

    Parameters
    ----------
    n: int
        The number of elements in the discretization.

    Returns
    -------
    _typing.sparray
        The 2D Poisson matrix of shape ((n - 1)**2, (n - 1)**2).
    """

    identity = sp.identity(n - 1)
    one_dimensional = get_inner_poisson_1D(n)

    return sparray(
        sp.kron(one_dimensional, identity) + sp.kron(identity, one_dimensional)
    )


def get_f_2D(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """It constructs the right-hand size in two dimensions.

    Parameters
    ----------
    n: int
        The number of discretization elements.

    Returns
    -------
    numpy.ndarray
        The right-hand side f of shape (n + 1)**2.
    numpy.ndarray
        The x-domain.
    numpy.ndarray
        The y-domain.
    """

    h = 1 / n
    space_domain = np.zeros((n + 1, n + 1))
    domain = np.linspace(0, 1, n + 1, endpoint=True)
    inner_domain = domain[1:-1]

    xx, yy = np.meshgrid(domain, domain, sparse=True)
    x_inner, y_inner = np.meshgrid(inner_domain, inner_domain, sparse=True)
    inner_f = f_2D(x_inner, y_inner)

    space_domain[1:-1, 1:-1] = inner_f
    space_domain[:, -1] = np.squeeze(np.sin(xx))
    space_domain[-1, :] = np.squeeze(np.sin(yy))

    # Add corrections
    space_domain[1, 1:-1] += space_domain[0, 1:-1] / h**2
    space_domain[-2, 1:-1] += space_domain[-1, 1:-1] / h**2
    space_domain[1:-1, 1] += space_domain[1:-1, 0] / h**2
    space_domain[1:-1, -2] += space_domain[1:-1, -1] / h**2

    return np.ravel(space_domain, order="C"), xx, yy


def f_2D(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The function to evaluate.

    Parameters
    ----------
    x: numpy.ndarray
        The x-domain.
    y: numpy.ndarray
        The y-domain.

    Returns
    -------
    numpy.ndarray
        The evaluation.
    """

    return (x**2 + y**2) * np.sin(x * y)


def get_exact_2D(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """It returns the exact solution to the Poisson equation on the grid.

    Parameters
    ----------
    n: int
        The number of discretization elements on each dimension.


    Returns
    -------
    numpy.ndarray
        The exact solution of shape (n + 1)**2.
    numpy.ndarray
        The x-domain.
    numpy.ndarray
        The y-domain.
    """

    domain = np.linspace(0, 1, n + 1, endpoint=True)
    xx, yy = np.meshgrid(domain, domain, sparse=True)
    return np.ravel(np.sin(np.multiply(xx, yy)), order="C"), xx, yy
