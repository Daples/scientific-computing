import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from poisson.two_dimensional import (
    get_exact_2D,
    get_f_2D,
    two_dimensional_poisson,
    _inner_no_boundary,
)
from poisson.three_dimensional import three_dimensional_poisson, _inner_no_boundary
from utils.solvers import solve_ssor_sparse
from utils.plotter import Plotter

omega = 1.5
N = 4
A = three_dimensional_poisson(N)
A_inner = _inner_no_boundary(N)
f, xx, yy = get_f_2D(N)
u_exact, _, _ = get_exact_2D(N)

plt.spy(A, color="k")
plt.savefig(Plotter.add_folder("full_A_3D.pdf"))

Plotter.clear()
plt.spy(A_inner, color="k")
plt.savefig(Plotter.add_folder("inner_A_3D.pdf"))

# results = solve_ssor_sparse(A, f, u_exact, omega)
# u = results.solution

# # Reshape
# u = u.reshape((N + 1, N + 1))
# u_exact = u_exact.reshape((N + 1, N + 1))

# fig, axs = plt.subplots(1, 2)
# im = axs[0].contourf(np.squeeze(xx), np.squeeze(yy), u)
# axs[0].set_title("Estimation")
# plt.colorbar(im, ax=axs[0])

# im = axs[1].contourf(np.squeeze(xx), np.squeeze(yy), u_exact)
# axs[1].set_title("Real")
# plt.colorbar(im, ax=axs[1])

# plt.show()

# Test reordering
# A = sp.csr_matrix(A)
# p = sp.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)

# I, J = np.ix_(p, p)
# App = A[I, J]

# Ap = A[p, :]
# Ap = Ap[:, p]

# App = A[p[:, None], p]

# fig, axs = plt.subplots(1, 3)
# axs[0].spy(A)
# axs[1].spy(Ap)
# axs[2].spy(App)
# plt.show()
