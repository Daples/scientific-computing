import numpy as np
import scipy.sparse as sp
from _typing import sparray

from poisson.one_dimensional import get_inner_poisson_1D
from utils import fill_block_matrix


def three_dimensional_poisson(n: int) -> sparray:
    """"""

    h = 1 / n
    diagonal_identity = sp.identity((n + 1) ** 2) * h**2
    rectangular_zeros = sp.lil_matrix(((n + 1) ** 2, (n + 1) ** 2 * (n - 1)))
    square_zeros = sp.lil_matrix(((n + 1) ** 2, (n + 1) ** 2))

    inner_poisson_matrix = get_inner_poisson_3D(n)
    poisson_matrix = sp.bmat(
        [
            [diagonal_identity, rectangular_zeros, square_zeros],
            [rectangular_zeros.T, inner_poisson_matrix, rectangular_zeros.T],
            [square_zeros, rectangular_zeros, diagonal_identity],
        ]
    ).tolil()

    return sparray(poisson_matrix / h**2)


def get_inner_poisson_3D(n: int) -> sparray:
    """Returns the inner Poisson matrix in 3D.

    Parameters
    ----------
    n: int
        The number of elements in the discretization on each dimension.

    Returns
    -------
    numpy.ndarray
        The inner matrix of shape ((n + 1)**2*(n - 1), (n + 1)**2*(n - 1)).
    """

    h = 1 / n
    dim = (n + 1) ** 2 * (n - 1)
    matrix = sp.lil_matrix((dim, dim))

    diagonal_blocks = n - 1
    subdiagonal_blocks = n - 2

    # Get Poisson with elimination of boundary conditions
    no_boundary_matrix = _inner_no_boundary(n)

    # Blocks on the main matrix to construct
    main_block_width = (n + 1) ** 2
    main_block_idx = np.array([0, main_block_width])

    # Inner blocks on main matrix
    inner_main_block_width = n + 1

    # Blocks on the auxiliary matrix with elimination of boundary conditions
    aux_block_width = (n - 1) ** 2
    aux_block_idx = np.array([0, aux_block_width])

    # Iterate over blocks on main matrix
    for i in range(diagonal_blocks):
        block = sp.lil_matrix((main_block_width, main_block_width))

        # Assign boundaries
        block[:inner_main_block_width, :inner_main_block_width] = (
            sp.identity(inner_main_block_width) * h**2
        )

        block[-inner_main_block_width:, -inner_main_block_width:] = (
            sp.identity(inner_main_block_width) * h**2
        )

        # Fill inner block matrix
        aux_idx = aux_block_idx + aux_block_width * i
        sub_matrix = sparray(
            no_boundary_matrix[
                aux_idx[0] : aux_idx[1],
                aux_idx[0] : aux_idx[1],
            ]
        )
        aux_matrix = fill_block_matrix(
            n,
            sub_matrix,
        )
        block[
            inner_main_block_width:-inner_main_block_width,
            inner_main_block_width:-inner_main_block_width,
        ] = aux_matrix

        # Assign block
        idx = main_block_idx + main_block_width * i
        matrix[idx[0] : idx[1], idx[0] : idx[1]] = block

        if i < subdiagonal_blocks:
            block = sp.lil_matrix((main_block_width, main_block_width))
            aux_matrix = fill_block_matrix(
                n, -sp.identity(aux_block_width, format="lil"), fill_subblocks=False
            )
            block[
                inner_main_block_width:-inner_main_block_width,
                inner_main_block_width:-inner_main_block_width,
            ] = aux_matrix
            shifted_idx = main_block_idx + main_block_width * (i + 1)

            # Lower
            matrix[shifted_idx[0] : shifted_idx[1], idx[0] : idx[1]] = block

            # Upper
            matrix[idx[0] : idx[1], shifted_idx[0] : shifted_idx[1]] = block
    return matrix


def _inner_no_boundary(n: int) -> sparray:
    """Returns a 3D Poisson matrix with elimination of boundary conditions.

    Parameters
    ----------
    n: int
        The number of elements in the discretization for each dimension.

    Returns
    -------
    numpy.ndarray
        The 2D Poisson matrix of shape ((n - 1)**3, (n - 1)**3).
    """

    one_dimensional = get_inner_poisson_1D(n)
    identity = sp.identity(n - 1, format="lil")
    t1 = sp.kron(
        sp.kron(identity, identity, format="lil"), one_dimensional, format="lil"
    )
    t2 = sp.kron(
        sp.kron(identity, one_dimensional, format="lil"), identity, format="lil"
    )
    t3 = sp.kron(
        sp.kron(one_dimensional, identity, format="lil"), identity, format="lil"
    )

    return sparray(t1 + t2 + t3)


def get_f_3D(n: int) -> np.ndarray:
    """"""

    h = 1 / n
    space_domain = np.zeros((n + 1, n + 1, n + 1))
    domain = np.linspace(0, 1, n + 1, endpoint=True)
    inner_domain = domain[1:-1]

    xx, yy, zz = np.meshgrid(domain, domain, domain, sparse=True)
    x_inner, y_inner, z_inner = np.meshgrid(
        inner_domain, inner_domain, inner_domain, sparse=True
    )
    inner_f = f_3D(x_inner, y_inner, z_inner)

    space_domain[1:-1, 1:-1, 1:-1] = inner_f
    space_domain[:, :, -1] = np.squeeze(np.sin(xx * yy))
    space_domain[:, -1, :] = np.squeeze(np.sin(xx * zz))
    space_domain[-1, :, :] = np.squeeze(np.sin(yy * zz))

    # Add corrections
    space_domain[1, 1:-1, 1:-1] += space_domain[0, 1:-1, 1:-1] / h**2
    space_domain[-2, 1:-1, 1:-1] += space_domain[-1, 1:-1, 1:-1] / h**2
    space_domain[1:-1, 1, 1:-1] += space_domain[1:-1, 0, 1:-1] / h**2
    space_domain[1:-1, -2, 1:-1] += space_domain[1:-1, -1, 1:-1] / h**2
    space_domain[1:-1, 1:-1, 1] += space_domain[1:-1, 1:-1, 0] / h**2
    space_domain[1:-1, 1:-1, -2] += space_domain[1:-1, 1:-1, -1] / h**2

    return np.ravel(space_domain, order="C")


def f_3D(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (x**2 * z**2 + z**2 * y**2 + y**2 * x**2) * np.sin(x * y * z)


def get_exact_3D(n: int) -> np.ndarray:
    domain = np.linspace(0, 1, n + 1, endpoint=True)
    xx, yy, zz = np.meshgrid(domain, domain, domain, sparse=True)
    return np.ravel(np.sin(np.multiply(xx, np.multiply(yy, zz))), order="C")
