import numpy as np
from typing import Callable

from scipy.sparse import diags
import matplotlib.pyplot as plt


def one_dimensional_poisson(n: int) -> np.ndarray:
    """"""

    h = 1 / n
    poisson_matrix = np.zeros((n + 1, n + 1))
    inner_poisson_matrix = get_inner_poisson_1D(n - 1)
    poisson_matrix[1:-1, 1:-1] = inner_poisson_matrix
    poisson_matrix[0, 0] = h**2
    poisson_matrix[-1, -1] = h**2

    return poisson_matrix


def get_inner_poisson_1D(n_inner: int) -> np.ndarray:
    """"""

    diagonals = [
        2 * np.ones(n_inner),
        -1 * np.ones(n_inner - 1),
        -1 * np.ones(n_inner - 1),
    ]
    inner_poisson_matrix = diags(diagonals, [0, -1, 1]).toarray()  # type: ignore

    return inner_poisson_matrix


def get_inner_poisson_2D(n_inner: int) -> np.ndarray:
    """"""

    aux_1D = get_inner_poisson_1D(n_inner)
    I = np.identity(n_inner)
    inner_poisson_matrix = np.kron(aux_1D, I) + np.kron(I, aux_1D)

    return inner_poisson_matrix


def three_dimensional_poisson(n: int) -> np.ndarray:
    """"""

    h = 1 / n
    poisson_matrix = np.identity((n + 1) ** 3) * h**2
    inner_poisson_matrix = get_inner_poisson_3D(n - 1)
    poisson_matrix[n:-n, n:-n] = inner_poisson_matrix

    return poisson_matrix


def get_inner_poisson_3D(n_inner: int) -> np.ndarray:
    """"""

    aux_2D = get_inner_poisson_2D(n_inner)
    I = np.identity(n_inner)
    inner_poisson_matrix = np.kron(aux_2D, I) + np.kron(I, aux_2D)

    return inner_poisson_matrix


def get_f(n: int, f: Callable) -> np.ndarray:
    """"""

    space_domain = np.zeros((n + 1, n + 1, n + 1))
    domain = np.linspace(0, 1, n + 1, endpoint=True)
    inner_domain = domain[1:-1]

    xx, yy, zz = np.meshgrid(domain, domain, domain, sparse=True)
    x_inner, y_inner, z_inner = np.meshgrid(
        inner_domain, inner_domain, inner_domain, sparse=True
    )
    inner_f = f(x_inner, y_inner, z_inner)

    space_domain[1:-1, 1:-1, 1:-1] = inner_f
    space_domain[:, :, -1] = np.squeeze(np.sin(xx * yy))
    space_domain[:, -1, :] = np.squeeze(np.sin(xx * zz))
    space_domain[-1, :, :] = np.squeeze(np.sin(yy * zz))

    return np.ravel(space_domain, order="C")


def solve_direct(n: int, f: Callable) -> np.ndarray:
    """"""

    A = three_dimensional_poisson(n)
    rhs = get_f(n, f)

    return np.linalg.solve(A, rhs)


f_3D = lambda x, y, z: (x**2 * z**2 + z**2 * y**2 + y**2 * x**2) * np.sin(
    x * y * z
)
print(solve_direct(4, f_3D))

# mat = get_inner_poisson_3D(5)
# print(mat.shape)
# # mat = np.linalg.inv(mat)
# plt.imshow(mat, interpolation="nearest")
# plt.colorbar()
# plt.show()
