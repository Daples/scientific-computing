import json
from time import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sla

from scipy.sparse.csgraph import reverse_cuthill_mckee
from tqdm import tqdm

from _typing import sparray
from poisson.two_dimensional import get_exact_2D, get_f_2D, two_dimensional_poisson
from utils import get_M_SSOR, solve_cg
from utils.plotter import Plotter
from utils.solvers import solve_ssor_sparse


# DIRECT SOLVERS
ps_2D = np.arange(2, 10)
Ns_2D = 2**ps_2D

for permute in [False, True]:
    params_times_2D = []
    factorization_times_2D = []
    solution_times_2D = []
    total_times_2D = []
    norms_2D = []
    ratios_2D = []

    perm_times_2D = []
    for N in tqdm(Ns_2D):
        total_time = time()

        # Get matrices
        params_time = time()
        A = two_dimensional_poisson(N)
        f, _, _ = get_f_2D(N)
        params_times_2D.append(time() - params_time)

        # Permute
        n = A.shape[0]
        perm = np.arange(n)
        if permute:
            perm_time = time()
            perm = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True)
            perm_times_2D.append(time() - perm_time)
            A = cast(sparray, A[perm[:, None], perm])
            f = f[perm]

        # Solve system
        fact_time = time()
        splu = sla.splu(A, permc_spec="NATURAL")
        factorization_times_2D.append(time() - fact_time)

        solve_time = time()
        u = splu.solve(f)
        solution_times_2D.append(time() - solve_time)

        # Undo permutation if necessary
        if permute:
            undo = np.zeros(n, dtype=int)
            undo[perm] = np.arange(n)
            u = u[undo]

        # Get results
        u_exact, _, _ = get_exact_2D(N)
        norms_2D.append(np.linalg.norm(u - u_exact, ord=np.inf))
        total_times_2D.append(time() - total_time)
        ratios_2D.append(splu.nnz / A.count_nonzero())

    results_2D = {
        "params_times": params_times_2D,
        "factorization_times": factorization_times_2D,
        "perm_times": perm_times_2D,
        "solution_times": solution_times_2D,
        "total_times": total_times_2D,
        "norms": norms_2D,
        "ratios": ratios_2D,
    }
    with open(f"results_chol_2D_{permute}.json", "w") as outfile:
        json.dump(results_2D, outfile)

    labels = [
        "Matrices Construction Time (s)",
        "Cholesky Factorization Time (s)",
        "Reverse-Cuthill McKee Time (s)",
        "Solution Time (s)",
        "Total time (s)",
        "$||u^h - u^h_{ex}||_\infty$",
        "Fill-in Ratio",
    ]
    for i, (key, val) in enumerate(results_2D.items()):
        path = key + f"_2D_{permute}.pdf"
        if len(val) > 0:
            Plotter.get_plot(ps_2D.tolist(), val, path, ylabel=labels[i])

# ITERATIVE SOLVERS
# Ns = [5, 15, 30, 45, 60, 75]
# omega = 1.5

# # SSOR as BIM
# d = {}
# results_per_N_SSOR = []
# for N in Ns:
#     # Get objects
#     A = two_dimensional_poisson(N)
#     f, xx, yy = get_f_2D(N)
#     u_exact, _, _ = get_exact_2D(N)

#     # Solve the system
#     results = solve_ssor_sparse(A, f, u_exact, omega)
#     results_per_N_SSOR.append(results)

# # Question 6
# fig, ax = plt.subplots(1, 1)
# for i, N in enumerate(Ns):
#     results = results_per_N_SSOR[i]
#     f, xx, yy = get_f_2D(N)
#     scaled_residuals = np.array(results.residuals) / np.linalg.norm(f)
#     ax.semilogy(scaled_residuals, label=f"N={N}")
# plt.xlabel("Iteration")
# plt.ylabel(r"$\log_{10}\frac{||r_m||_2}{||f^h||_2}$")

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# plt.grid(True, which="both", ls="-")
# plt.savefig(Plotter.add_folder(f"residuals_SSOR_2D.pdf"), bbox_inches="tight")

# d["scaled_residuals"] = [r.residuals for r in results_per_N_SSOR]

# # Question 7
# last_iters = 5
# residual_reduction = np.zeros((len(Ns), last_iters))
# for i, N in enumerate(Ns):
#     residuals = np.array(results_per_N_SSOR[i].residuals)
#     residual_reduction[i, :] = np.divide(
#         residuals[-last_iters:], residuals[-last_iters - 1 : -1]
#     )
# np.savetxt("asymptotic_convergence.csv", residual_reduction, delimiter=",")
# d["asymptotic_convergence"] = residual_reduction.tolist()

# # Question 8
# cpu_times = [r.time for r in results_per_N_SSOR]
# Plotter.get_plot(
#     cast(list[float], Ns),
#     cpu_times,
#     "cpu_SSOR_2D.pdf",
#     xlabel="$N$",
#     ylabel="CPU Time (s)",
# )
# d["cpu_times"] = cpu_times

# with open("results_SSOR_2D.json", "w") as outfile:
#     json.dump(d, outfile)

# # CG with SSOR preconditioner
# d = {}
# results_per_N_CG = []
# for N in Ns:
#     # Get objects
#     A = two_dimensional_poisson(N)
#     f, _, _ = get_f_2D(N)
#     u_exact, _, _ = get_exact_2D(N)

#     # Get preconditioner
#     M = get_M_SSOR(A, omega)

#     # Solve system
#     results = solve_cg(A, f, u_exact, M=M)
#     results_per_N_CG.append(results)

# # Question 9
# fig, ax = plt.subplots(1, 1)
# for i, N in enumerate(Ns):
#     results = results_per_N_CG[i]
#     f, xx, yy = get_f_2D(N)
#     scaled_residuals = np.array(results.residuals) / np.linalg.norm(f)
#     ax.semilogy(scaled_residuals, label=f"N={N}")
# plt.xlabel("Iteration")
# plt.ylabel(r"$\log_{10}\frac{||r_m||_2}{||f^h||_2}$")

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# plt.grid(True, which="both", ls="-")
# plt.savefig(Plotter.add_folder(f"residuals_CG_2D.pdf"), bbox_inches="tight")

# d["scaled_residuals"] = [r.residuals for r in results_per_N_CG]

# cpu_times = [r.time for r in results_per_N_CG]
# Plotter.get_plot(
#     cast(list[float], Ns),
#     cpu_times,
#     "cpu_CG_2D.pdf",
#     xlabel="$N$",
#     ylabel="CPU Time (s)",
# )
# d["cpu_times"] = cpu_times

# with open("results_CG_2D.json", "w") as outfile:
#     json.dump(d, outfile)
