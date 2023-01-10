import numpy as np
import scipy.sparse.linalg as sla
from tqdm import tqdm

from poisson.two_dimensional import two_dimensional_poisson, get_f_2D, get_exact_2D
from poisson.three_dimensional import three_dimensional_poisson, get_f_3D, get_exact_3D

Ns_2D = 2 ** np.arange(2, 6)
Ns_3D = 2 ** np.arange(2, 4)

# DIRECT SOLVERS

norms_2D = []
for N in tqdm(Ns_2D):
    # Get matrices
    A = two_dimensional_poisson(N)
    f = get_f_2D(N)

    # Solve system
    u = sla.spsolve(A, f)

    # Get results
    u_exact = get_exact_2D(N)
    norms_2D.append(np.linalg.norm(u - u_exact, ord=np.inf))

print(norms_2D)

# norms_3D = []
# for N in tqdm(Ns_3D):
#     # Get matrices
#     A = three_dimensional_poisson(N)
#     f = get_f_3D(N)

#     # Solve system
#     u = sla.spsolve(A, f)

#     # Get results
#     u_exact = get_exact_3D(N)
#     norms_3D.append(np.linalg.norm(u - u_exact, ord=np.inf))

# print(norms_3D)
