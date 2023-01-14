# import json
# from time import time
# from typing import cast

# import numpy as np
# import scipy.sparse.linalg as sla
# from scipy.sparse.csgraph import reverse_cuthill_mckee
# from tqdm import tqdm

# from _typing import sparray
# from poisson.three_dimensional import get_exact_3D, get_f_3D, three_dimensional_poisson

# from poisson.two_dimensional import get_exact_2D, get_f_2D, two_dimensional_poisson
# from utils import get_M_SSOR, solve_cg
# from utils.plotter import Plotter

# ps_2D = np.arange(2, 11)
# Ns_2D = 2**ps_2D
# ps_3D = np.arange(2, 7)
# Ns_3D = 2**ps_3D

# # DIRECT SOLVERS
# params_times_2D = []
# factorization_times_2D = []
# solution_times_2D = []
# total_times_2D = []
# norms_2D = []
# ratios_2D = []

# permute = False
# perm_times_2D = []

# for N in tqdm(Ns_2D):
#     total_time = time()

#     # Get matrices
#     params_time = time()
#     A = two_dimensional_poisson(N)
#     f, _, _ = get_f_2D(N)
#     params_times_2D.append(time() - params_time)

#     # Permute
#     n = A.shape[0]
#     perm = np.arange(n)
#     if permute:
#         perm_time = time()
#         perm = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True)
#         perm_times_2D.append(time() - perm_time)
#         idx_i, idx_j = np.ix_(perm, perm)
#         A = cast(sparray, A[idx_i, idx_j])
#         f = f[perm]

#     # Solve system
#     fact_time = time()
#     splu = sla.splu(A)
#     factorization_times_2D.append(time() - fact_time)

#     solve_time = time()
#     u = splu.solve(f)
#     solution_times_2D.append(time() - solve_time)

#     # Undo permutation if necessary
#     if permute:
#         undo = np.zeros(n, dtype=int)
#         undo[perm] = np.arange(n)
#         u = u[undo]

#     # Get results
#     u_exact, _, _ = get_exact_2D(N)
#     norms_2D.append(np.linalg.norm(u - u_exact, ord=np.inf))
#     total_times_2D.append(time() - total_time)
#     ratios_2D.append(splu.nnz / A.count_nonzero())

# results_2D = {
#     "params_times": params_times_2D,
#     "factorization_times": factorization_times_2D,
#     "perm_times": perm_times_2D,
#     "solution_times": solution_times_2D,
#     "total_times": total_times_2D,
#     "norms": norms_2D,
#     "ratios": ratios_2D,
# }
# with open("results_chol_2D.json", "w") as outfile:
#     json.dump(results_2D, outfile)

# labels = [
#     "Matrices Construction Time (s)",
#     "Cholesky Factorization Time (s)",
#     "Reverse-Cuthill McKee Time (s)",
#     "Solution Time (s)",
#     "Total time (s)",
#     "$||u^h - u^h_{ex}||_\infty$",
#     "Fill-in Ratio",
# ]
# for i, (key, val) in enumerate(results_2D.items()):
#     path = key + "_2D_no_perm.pdf"
#     if len(val) > 0:
#         Plotter.get_plot(ps_2D.tolist(), val, path, ylabel=labels[i])


# params_times_3D = []
# factorization_times_3D = []
# solution_times_3D = []
# total_times_3D = []
# norms_3D = []
# ratios_3D = []

# permute = True
# perm_times_3D = []

# for N in tqdm(Ns_3D):
#     total_time = time()

#     # Get matrices
#     params_time = time()
#     A = three_dimensional_poisson(N)
#     f, _, _ = get_f_3D(N)
#     params_times_3D.append(time() - params_time)

#     # Permute
#     n = A.shape[0]
#     perm = np.arange(n)
#     if permute:
#         perm_time = time()
#         perm = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True)
#         perm_times_3D.append(time() - perm_time)
#         idx_i, idx_j = np.ix_(perm, perm)
#         A = cast(sparray, A[idx_i, idx_j])
#         f = f[perm]

#     # Solve system
#     fact_time = time()
#     splu = sla.splu(A)
#     factorization_times_3D.append(time() - fact_time)

#     solve_time = time()
#     u = splu.solve(f)
#     solution_times_3D.append(time() - solve_time)

#     # Undo permutation if necessary
#     if permute:
#         undo = np.zeros(n, dtype=int)
#         undo[perm] = np.arange(n)
#         u = u[undo]

#     # Get results
#     u_exact, _, _ = get_exact_3D(N)
#     norms_3D.append(np.linalg.norm(u - u_exact, ord=np.inf))
#     total_times_3D.append(time() - total_time)
#     ratios_3D.append(splu.nnz / A.count_nonzero())

# results_3D = {
#     "params_times": params_times_3D,
#     "factorization_times": factorization_times_3D,
#     "perm_times": perm_times_3D,
#     "solution_times": solution_times_3D,
#     "total_times": total_times_3D,
#     "norms": norms_3D,
#     "ratios": ratios_3D,
# }
# with open("results_chol_3D_perm.json", "w") as outfile:
#     json.dump(results_3D, outfile)

# labels = [
#     "Matrices Construction Time (s)",
#     "Cholesky Factorization Time (s)",
#     "Reverse-Cuthill McKee Time (s)",
#     "Solution Time (s)",
#     "Total time (s)",
#     "$||u^h - u^h_{ex}||_\infty$",
#     "Fill-in Ratio",
# ]
# for i, (key, val) in enumerate(results_3D.items()):
#     path = key + "_3D_perm.pdf"
#     Plotter.get_plot(ps_3D.tolist(), val, path, ylabel=labels[i])

# ITERATIVE SOLVERS
# SSOR as BIM

# CG with SSOR preconditioner
# Ns = [5, 25, 50, 75, 100, 150, 200]
# Ns = [5, 25]
# omega = 1.5
# for N in Ns:
#     # Get objects
#     A = two_dimensional_poisson(N)
#     f = get_f_2D(N)
#     u_exact = get_exact_2D(N)

#     # Get preconditioner
#     M = get_M_SSOR(A, omega)

#     # Solve system
#     results = solve_cg(A, f, u_exact, M=M)

# a = 1
