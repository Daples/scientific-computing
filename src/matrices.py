from typing import cast
import numpy as np
import scipy.sparse.linalg as sla
import json
from tqdm import tqdm
from time import time
from scipy.sparse.csgraph import reverse_cuthill_mckee
from _typing import sparray

from poisson.two_dimensional import two_dimensional_poisson, get_f_2D, get_exact_2D
from poisson.three_dimensional import three_dimensional_poisson, get_f_3D, get_exact_3D
from utils.plotter import Plotter

ps_2D = np.arange(2, 4)
Ns_2D = 2**ps_2D
ps_3D = np.arange(2, 7)
Ns_3D = 2**ps_3D

# DIRECT SOLVERS
params_times_2D = []
factorization_times_2D = []
solution_times_2D = []
total_times_2D = []
norms_2D = []
ratios_2D = []
permute = True

for N in tqdm(Ns_2D):
    total_time = time()

    # Get matrices
    params_time = time()
    A = two_dimensional_poisson(N)
    f = get_f_2D(N)
    params_times_2D.append(time() - params_time)

    # Permute
    if permute:
        perm = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True)
        A = cast(sparray, A[np.ix_(perm, perm)])

    # Solve system
    fact_time = time()
    splu = sla.splu(A)
    factorization_times_2D.append(time() - fact_time)

    solve_time = time()
    u = splu.solve(f)
    solution_times_2D.append(time() - solve_time)

    # Get results
    u_exact = get_exact_2D(N)
    norms_2D.append(np.linalg.norm(u - u_exact, ord=np.inf))
    total_times_2D.append(time() - total_time)
    ratios_2D.append(splu.nnz / A.count_nonzero())

results_2D = {
    "params_times": params_times_2D,
    "factorization_times": factorization_times_2D,
    "solution_times": solution_times_2D,
    "total_times": total_times_2D,
    "norms": norms_2D,
    "ratios": ratios_2D,
}
with open("results_2D.json", "w") as outfile:
    json.dump(results_2D, outfile)

labels = [
    "Matrices Construction Time (s)",
    "Cholesky Factorization Time (s)",
    "Solution Time (s)",
    "Total time (s)",
    "$||u^h - u^h_{ex}||_\infty$",
    "Fill-in Ratio",
]
for i, (key, val) in enumerate(results_2D.items()):
    path = key + "_2D.pdf"
    Plotter.get_plot(ps_2D.tolist(), val, path, ylabel=labels[i])


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
