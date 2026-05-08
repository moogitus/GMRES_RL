"""
Shared helpers for matrix loading, problem validation, the SB3 callback
used by the DQN drivers, and per-seed run aggregation.

The SB3 callback subclasses ``BaseCallback`` from stable-baselines3
(https://github.com/DLR-RM/stable-baselines3).
"""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from stable_baselines3.common.callbacks import BaseCallback


def _largest_matrix_member(tar: tarfile.TarFile):
    members = [
        member for member in tar.getmembers()
        if member.name.endswith(".mtx") and not member.name.endswith("_b.mtx")
    ]
    if not members:
        raise ValueError("No matrix .mtx file found.")
    return max(members, key=lambda member: member.size)


def _flatten_rhs(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _load_mtx_file(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return mmread(io.BytesIO(handle.read()))


def _validate_problem(name: str, A, b):
    A = csr_matrix(A.astype(np.float64))
    b = _flatten_rhs(b)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: matrix is not square")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"{name}: RHS dimension mismatch")
    return A, b


def _consistent_rhs(A) -> np.ndarray:
    x_true = np.ones(A.shape[1], dtype=np.float64)
    return np.asarray(A @ x_true, dtype=np.float64).reshape(-1)


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _recursive_candidates(root: Path, patterns):
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
    return candidates


def _find_local_archive(name: str, matrices_dir: Path):
    direct = _first_existing([
        matrices_dir / f"{name}.tar.gz",
        matrices_dir / f"{name}.tgz",
        matrices_dir / name / f"{name}.tar.gz",
        matrices_dir / name / f"{name}.tgz",
    ])
    if direct is not None:
        return direct

    matches = _recursive_candidates(matrices_dir, [f"{name}.tar.gz", f"{name}.tgz"])
    return matches[0] if matches else None


# sb3 callback that records the relative-residual norm and chosen m
# at every env step, and halts the rollout once the env signals done.
class RelativeResidualCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.relative_residuals = []
        self.ms = []
        self._done = False

    def _on_step(self):
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            if "relative_residual_norm" in info:
                self.relative_residuals.append(float(info["relative_residual_norm"]))
            if "current_m" in info:
                self.ms.append(int(info["current_m"]))
            if done:
                self._done = True
        return not self._done


def summarise_runs(runs):
    def _avg(key):
        return float(np.mean([run[key] for run in runs]))

    def _std(key):
        return float(np.std([run[key] for run in runs]))

    return {
        "convergence_rate": _avg("converged"),
        "arnoldi_mean": _avg("total_arnoldi"),
        "arnoldi_std": _std("total_arnoldi"),
        "time_mean": _avg("elapsed_seconds"),
        "time_std": _std("elapsed_seconds"),
        "cycles_mean": _avg("cycles_to_tol"),
        "final_residual_norm_mean": _avg("final_residual_norm"),
        "final_relative_residual_norm_mean": _avg("final_relative_residual_norm"),
    }
