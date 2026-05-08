"""
Generates the 20 1D convection-diffusion matrices used in §4.3
(κ-sweep). Each matrix discretizes -ε u'' + β u' = f on [0,1] with
Dirichlet boundaries via central FD; sweeping (ε, n) varies κ_2(A) over
orders of magnitude with the generating PDE held fixed.

Outputs Matrix Market (.mtx) files under matrices/kappa_sweep_20_matrices/.
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.io import mmwrite
from pathlib import Path


# tridiagonal upwind-stabilised central FD for -eps u'' + beta u' = f on
# [0,1] with Dirichlet boundaries; n interior nodes, h = 1/(n+1)
def make_convdiff_1d_sparse(n: int, eps: float, beta: float = 1.0):
    h = 1.0 / (n + 1)
    diag_v = np.full(n,     2 * eps / h**2)
    lo_v   = np.full(n - 1, -eps / h**2 - beta / (2 * h))
    hi_v   = np.full(n - 1, -eps / h**2 + beta / (2 * h))
    return csr_matrix(diags([lo_v, diag_v, hi_v], [-1, 0, 1]))


MATRICES = [
    {"n": 1000, "eps": 0.05, "name": "convdiff_n1000_eps05_easy"},
    {"n": 1000, "eps": 0.10, "name": "convdiff_n1000_eps10_medium"},
    {"n": 1000, "eps": 0.15, "name": "convdiff_n1000_eps15_hard"},
    {"n": 1500, "eps": 0.15, "name": "convdiff_n1500_eps15_very_hard"},
    {"n": 2000, "eps": 0.15, "name": "convdiff_n2000_eps15_extreme"},
    {"n":  500, "eps": 0.05, "name": "convdiff_n500_eps05"},
    {"n":  700, "eps": 0.05, "name": "convdiff_n700_eps05"},
    {"n":  500, "eps": 0.10, "name": "convdiff_n500_eps10"},
    {"n":  600, "eps": 0.15, "name": "convdiff_n600_eps15"},
    {"n":  800, "eps": 0.15, "name": "convdiff_n800_eps15"},
    {"n": 1500, "eps": 0.07, "name": "convdiff_n1500_eps07"},
    {"n":  900, "eps": 0.20, "name": "convdiff_n900_eps20"},
    {"n": 1800, "eps": 0.08, "name": "convdiff_n1800_eps08"},
    {"n": 1100, "eps": 0.25, "name": "convdiff_n1100_eps25"},
    {"n": 2000, "eps": 0.10, "name": "convdiff_n2000_eps10"},
    {"n": 2500, "eps": 0.09, "name": "convdiff_n2500_eps09"},
    {"n": 3500, "eps": 0.07, "name": "convdiff_n3500_eps07"},
    {"n": 2200, "eps": 0.18, "name": "convdiff_n2200_eps18"},
    {"n": 2500, "eps": 0.18, "name": "convdiff_n2500_eps18"},
    {"n": 2800, "eps": 0.20, "name": "convdiff_n2800_eps20"},
]

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "matrices" / "kappa_sweep_20_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg in MATRICES:
        A = make_convdiff_1d_sparse(cfg["n"], cfg["eps"])
        path = out_dir / f"{cfg['name']}.mtx"
        mmwrite(str(path), A)
        print(f"Saved {path.name}  shape={A.shape}  nnz={A.nnz}")
