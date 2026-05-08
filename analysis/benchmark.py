"""
Run a comprehensive benchmark over the 103-matrix test suite using:
  - GMRES(20)
  - GMRES(60)
  - angleGMRES
  - randGMRES
  - DQN-controlled GMRES_RL (history length 5, m_max 20 by default)

All methods use the consistent RHS b = A @ 1 and a common total-Arnoldi budget.
"""

import argparse
import concurrent.futures
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from scipy.sparse import csr_matrix
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from env import GMRESEnv
from train_dqn import discover_matrix_names, load_problem


class BudgetedGMRESEnv(GMRESEnv):
    """GMRES environment with an optional custom action menu and Arnoldi cap."""

    def __init__(
        self,
        *,
        action_values: list[int] | tuple[int, ...] | None = None,
        max_total_arnoldi: int | None = None,
        **kwargs,
    ):
        if action_values is None:
            m_max = int(kwargs["m_max"])
            action_values = list(range(1, m_max + 1))
        else:
            action_values = [int(m) for m in action_values]
            kwargs["m_max"] = max(action_values)

        self.max_total_arnoldi = (
            None if max_total_arnoldi is None else int(max_total_arnoldi)
        )
        super().__init__(**kwargs)
        self.action_ms = tuple(action_values)
        self.action_space = spaces.Discrete(len(self.action_ms))

    def step(self, action):
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.action_ms):
            raise ValueError(f"action index {action_idx} outside valid range")

        m = int(self.action_ms[action_idx])
        prev_norm = self.current_residual_norm
        prev_rel = self._relative_residual(prev_norm)

        if self._is_converged(prev_norm, prev_rel):
            obs = self._build_observation()
            return obs, 0.0, True, False, {
                "current_m": m,
                "residual_norm": prev_norm,
                "relative_residual_norm": prev_rel,
                "cycle_count": self.cycle_count,
                "total_arnoldi": self.total_arnoldi,
            }

        self.x, self.current_residual_vector, actual_m = self._execute_gmres_cycle(
            self.x, self.current_residual_vector, m
        )
        self.current_residual_norm = float(np.linalg.norm(self.current_residual_vector))
        self.cycle_count += 1
        self.total_arnoldi += actual_m

        curr_rel = self._relative_residual(self.current_residual_norm)
        reward = self._compute_reward(
            prev_norm, prev_rel, self.current_residual_norm, curr_rel, m
        )

        self.restart_history = np.roll(self.restart_history, -1)
        self.restart_history[-1] = np.float32(m / self.m_max)
        self.residual_history = np.roll(self.residual_history, -1)
        self.residual_history[-1] = np.float32(np.log(max(curr_rel, 1e-12)))

        terminated = bool(self._is_converged(self.current_residual_norm, curr_rel))
        truncated = bool(self.cycle_count >= self.max_cycles) and not terminated
        if (
            self.max_total_arnoldi is not None
            and self.total_arnoldi >= self.max_total_arnoldi
            and not terminated
        ):
            truncated = True

        obs = self._build_observation()
        info = {
            "current_m": m,
            "actual_arnoldi": int(actual_m),
            "residual_norm": self.current_residual_norm,
            "prev_residual_norm": prev_norm,
            "relative_residual_norm": curr_rel,
            "prev_relative_residual_norm": prev_rel,
            "cycle_count": self.cycle_count,
            "total_arnoldi": self.total_arnoldi,
        }
        return obs, float(reward), terminated, truncated, info


class _TraceOnDone(BaseCallback):
    def __init__(self):
        super().__init__()
        self._done = False
        self._t0 = None
        self.done_index: int | None = None
        self.residual_norms: list[float] = []
        self.relative_residuals: list[float] = []
        self.ms: list[int] = []
        self.actual_arnoldi: list[int] = []
        self.step_times: list[float] = []

    def start_timer(self):
        self._t0 = time.perf_counter()

    def _on_step(self):
        if self._done:
            return False
        if self._t0 is None:
            self.start_timer()
        now = time.perf_counter() - self._t0
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            self.residual_norms.append(float(info.get("residual_norm", np.nan)))
            self.relative_residuals.append(
                float(info.get("relative_residual_norm", np.nan))
            )
            self.ms.append(int(info.get("current_m", 0)))
            self.actual_arnoldi.append(int(info.get("actual_arnoldi", 0)))
            self.step_times.append(float(now))
            if done:
                self.done_index = len(self.ms)
                self._done = True
                break
        return not self._done


def _trace_payload(initial_residual_norm: float, initial_relative_residual: float, logger: _TraceOnDone) -> dict:
    end = logger.done_index if logger.done_index is not None else len(logger.ms)
    ms = np.asarray(logger.ms[:end], dtype=np.int64)
    actual_arnoldi = np.asarray(logger.actual_arnoldi[:end], dtype=np.int64)
    arnoldi = np.concatenate([[0], np.cumsum(actual_arnoldi, dtype=np.int64)])
    wallclock = np.concatenate([[0.0], np.asarray(logger.step_times[:end], dtype=np.float64)])
    residual_norm = np.concatenate(
        [[initial_residual_norm], np.asarray(logger.residual_norms[:end], dtype=np.float64)]
    )
    relative_residual = np.concatenate(
        [[initial_relative_residual], np.asarray(logger.relative_residuals[:end], dtype=np.float64)]
    )
    return {
        "arnoldi_steps": arnoldi.astype(np.int64).tolist(),
        "wallclock_seconds": wallclock.astype(np.float64).tolist(),
        "residual_norm": residual_norm.astype(np.float64).tolist(),
        "relative_residual_norm": relative_residual.astype(np.float64).tolist(),
        "restart_values": ms.astype(np.int64).tolist(),
        "actual_arnoldi": actual_arnoldi.astype(np.int64).tolist(),
    }


def _run_policy_trace(env: BudgetedGMRESEnv, action_fn) -> dict:
    _, reset_info = env.reset()
    arnoldi = [0]
    wallclock = [0.0]
    residual_norm = [float(reset_info["residual_norm"])]
    relative_residual = [float(reset_info["relative_residual_norm"])]
    restart_values = []
    t0 = time.perf_counter()
    info = dict(reset_info)

    while True:
        action = int(action_fn(info))
        _, _, terminated, truncated, info = env.step(action)
        restart_values.append(int(info["current_m"]))
        arnoldi.append(int(info["total_arnoldi"]))
        wallclock.append(float(time.perf_counter() - t0))
        residual_norm.append(float(info["residual_norm"]))
        relative_residual.append(float(info["relative_residual_norm"]))
        if terminated or truncated:
            break

    return {
        "converged": bool(env._is_converged(residual_norm[-1], relative_residual[-1])),
        "cycles_to_tol": int(info["cycle_count"]),
        "total_arnoldi": int(arnoldi[-1]),
        "elapsed_seconds": float(wallclock[-1]),
        "final_residual_norm": float(residual_norm[-1]),
        "final_relative_residual_norm": float(relative_residual[-1]),
        "trace": {
            "arnoldi_steps": arnoldi,
            "wallclock_seconds": wallclock,
            "residual_norm": residual_norm,
            "relative_residual_norm": relative_residual,
            "restart_values": restart_values,
        },
    }


def run_fixed_gmres(
    A,
    b,
    restart: int,
    tolerance: float,
    absolute_tolerance: float,
    max_total_arnoldi: int,
) -> dict:
    max_cycles = int(math.ceil(max_total_arnoldi / restart))
    env = BudgetedGMRESEnv(
        A=csr_matrix(A),
        b=b,
        action_values=[restart],
        m_max=restart,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        max_cycles=max_cycles,
        max_total_arnoldi=max_total_arnoldi,
        history_length=1,
        lambda_work=0.0,
        gamma_shape=1.0,
        convergence_bonus=0.0,
    )
    return _run_policy_trace(env, lambda info: 0)


def run_rand_gmres(
    A,
    b,
    restart_values: list[int],
    tolerance: float,
    absolute_tolerance: float,
    max_total_arnoldi: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    max_cycles = int(math.ceil(max_total_arnoldi / min(restart_values)))
    env = BudgetedGMRESEnv(
        A=csr_matrix(A),
        b=b,
        action_values=restart_values,
        m_max=max(restart_values),
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        max_cycles=max_cycles,
        max_total_arnoldi=max_total_arnoldi,
        history_length=1,
        lambda_work=0.0,
        gamma_shape=1.0,
        convergence_bonus=0.0,
    )
    return _run_policy_trace(env, lambda info: int(rng.integers(len(restart_values))))


def run_angle_gmres(
    A,
    b,
    m_max: int,
    m_min: int,
    decrement: int,
    tolerance: float,
    absolute_tolerance: float,
    max_total_arnoldi: int,
    beta_small: float,
    beta_large: float,
) -> dict:
    m_max = int(m_max)
    m_min = int(m_min)
    decrement = int(decrement)
    if decrement <= 0:
        raise ValueError("angleGMRES decrement must be positive")
    if m_min < 1:
        raise ValueError("angleGMRES m_min must be at least 1")
    if m_max <= m_min:
        raise ValueError("angleGMRES requires m_max > m_min")

    ordered = [m_max]
    current = m_max
    while current > m_min:
        current = max(m_min, current - decrement)
        ordered.append(current)

    max_cycles = int(math.ceil(max_total_arnoldi / m_min))
    env = BudgetedGMRESEnv(
        A=csr_matrix(A),
        b=b,
        action_values=ordered,
        m_max=m_max,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        max_cycles=max_cycles,
        max_total_arnoldi=max_total_arnoldi,
        history_length=1,
        lambda_work=0.0,
        gamma_shape=1.0,
        convergence_bonus=0.0,
    )
    state = {"idx": 0}

    def policy(info):
        action = state["idx"]
        if "prev_relative_residual_norm" in info and "relative_residual_norm" in info:
            prev_rel = max(float(info["prev_relative_residual_norm"]), 1e-16)
            curr_rel = float(info["relative_residual_norm"])
            ratio = curr_rel / prev_rel
            current_m = ordered[state["idx"]]

            # Paper's alpha-GMRES rule:
            # - if convergence is very good, keep the same restart
            # - if convergence is poor, reset to m_max
            # - otherwise decrement by d until m_min, then reset to m_max
            if ratio < beta_small:
                next_m = current_m
            elif ratio > beta_large:
                next_m = m_max
            elif current_m > m_min:
                next_m = max(m_min, current_m - decrement)
            else:
                next_m = m_max

            state["idx"] = ordered.index(next_m)
        return action

    return _run_policy_trace(env, policy)


def run_dqn_gmres(
    A,
    b,
    *,
    m_max: int,
    history_length: int,
    tolerance: float,
    absolute_tolerance: float,
    max_total_arnoldi: int,
    gamma: float,
    lambda_work: float,
    convergence_bonus: float,
    learning_rate: float,
    buffer_size: int,
    learning_starts: int,
    batch_size: int,
    target_update_interval: int,
    exploration_fraction: float,
    exploration_final_eps: float,
    device: str,
    seed: int,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    max_cycles = int(max_total_arnoldi)
    env = BudgetedGMRESEnv(
        A=A,
        b=b,
        m_max=m_max,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        max_cycles=max_cycles,
        max_total_arnoldi=max_total_arnoldi,
        history_length=history_length,
        lambda_work=lambda_work,
        gamma_shape=gamma,
        convergence_bonus=convergence_bonus,
    )
    _, reset_info = env.reset(seed=seed)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=seed,
        device=device,
    )

    logger = _TraceOnDone()
    logger.start_timer()
    model.learn(total_timesteps=max_cycles, callback=logger, progress_bar=False)

    trace = _trace_payload(
        initial_residual_norm=float(reset_info["residual_norm"]),
        initial_relative_residual=float(reset_info["relative_residual_norm"]),
        logger=logger,
    )
    final_residual = float(trace["residual_norm"][-1])
    final_rel = float(trace["relative_residual_norm"][-1])
    return {
        "converged": bool(env._is_converged(final_residual, final_rel)),
        "cycles_to_tol": max(0, len(trace["arnoldi_steps"]) - 1),
        "total_arnoldi": int(trace["arnoldi_steps"][-1]),
        "elapsed_seconds": float(trace["wallclock_seconds"][-1]),
        "final_residual_norm": final_residual,
        "final_relative_residual_norm": final_rel,
        "trace": trace,
    }


def summarise_runs(runs: list[dict]) -> dict:
    return {
        "convergence_rate": float(np.mean([run["converged"] for run in runs])),
        "arnoldi_mean": float(np.mean([run["total_arnoldi"] for run in runs])),
        "arnoldi_std": float(np.std([run["total_arnoldi"] for run in runs])),
        "time_mean": float(np.mean([run["elapsed_seconds"] for run in runs])),
        "time_std": float(np.std([run["elapsed_seconds"] for run in runs])),
        "final_relative_residual_norm_mean": float(
            np.mean([run["final_relative_residual_norm"] for run in runs])
        ),
    }


def _method_specs(args: argparse.Namespace) -> dict[str, dict]:
    def restart_label(prefix: str, values: list[int]) -> str:
        values_text = ",".join(str(value) for value in values)
        return f"{prefix}({values_text})"

    def angle_label(m_max: int, decrement: int, m_min: int) -> str:
        return f"angleGMRES(max={m_max},d={decrement},min={m_min})"

    return {
        "gmres20": {
            "label": "GMRES(20)",
            "type": "fixed",
            "restart": args.gmres20_restart,
            "seeds": False,
        },
        "gmres60": {
            "label": "GMRES(60)",
            "type": "fixed",
            "restart": args.gmres60_restart,
            "seeds": False,
        },
        "rand_gmres_10_20": {
            "label": restart_label("randGMRES", list(args.rand_restarts_short)),
            "type": "rand",
            "restart_values": list(args.rand_restarts_short),
            "seeds": True,
        },
        "rand_gmres_10_20_30_40_50_60": {
            "label": restart_label("randGMRES", list(args.rand_restarts_wide)),
            "type": "rand",
            "restart_values": list(args.rand_restarts_wide),
            "seeds": True,
        },
        "gmres_rl_m20": {
            "label": "GMRES_RL(max=20)",
            "type": "dqn",
            "m_max": 20,
            "seeds": True,
        },
        "gmres_rl_m60": {
            "label": "GMRES_RL(max=60)",
            "type": "dqn",
            "m_max": 60,
            "seeds": True,
        },
        "angle_gmres_10_20": {
            "label": angle_label(
                args.angle_short_max,
                args.angle_short_decrement,
                args.angle_short_min,
            ),
            "type": "angle",
            "m_max": args.angle_short_max,
            "m_min": args.angle_short_min,
            "decrement": args.angle_short_decrement,
            "seeds": False,
        },
        "angle_gmres_10_20_30_40_50_60": {
            "label": angle_label(
                args.angle_wide_max,
                args.angle_wide_decrement,
                args.angle_wide_min,
            ),
            "type": "angle",
            "m_max": args.angle_wide_max,
            "m_min": args.angle_wide_min,
            "decrement": args.angle_wide_decrement,
            "seeds": False,
        },
    }


def _build_payload(
    *,
    matrices_root: Path,
    names: list[str],
    args: argparse.Namespace,
    results: dict,
    run_complete: bool,
) -> dict:
    return {
        "meta": {
            "matrices_dir": str(matrices_root),
            "matrix_names": names,
            "tolerance": args.tolerance,
            "max_total_arnoldi": args.max_total_arnoldi,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "device": args.device,
            "n_jobs": args.n_jobs,
            "methods": list(args.methods),
            "method_params": {
                "gmres20_restart": args.gmres20_restart,
                "gmres60_restart": args.gmres60_restart,
                "absolute_tolerance": args.absolute_tolerance,
                "rand_restarts_short": list(args.rand_restarts_short),
                "rand_restarts_wide": list(args.rand_restarts_wide),
                "angle_short_max": args.angle_short_max,
                "angle_short_min": args.angle_short_min,
                "angle_short_decrement": args.angle_short_decrement,
                "angle_wide_max": args.angle_wide_max,
                "angle_wide_min": args.angle_wide_min,
                "angle_wide_decrement": args.angle_wide_decrement,
                "angle_beta_small": args.angle_beta_small,
                "angle_beta_large": args.angle_beta_large,
                "dqn_history_length": args.dqn_history_length,
                "dqn_gamma": args.dqn_gamma,
                "dqn_lambda_work": args.dqn_lambda_work,
                "dqn_convergence_bonus": args.dqn_convergence_bonus,
            },
            "notes": [
                "All methods use the consistent RHS b = A @ 1.",
                "A common total-Arnoldi budget is enforced across methods.",
                "GMRES_RL uses the repo's DQN with history-length state and the configured m_max values.",
                "angleGMRES implements the alpha-GMRES residual-ratio rule from Baker, Jessup, and Kolev (2009): keep m on very good progress, reset to m_max on poor progress, otherwise decrement by d until m_min before cycling back to m_max.",
                "randGMRES samples uniformly from the configured restart lists at each restart cycle.",
                "A partial checkpoint JSON is written after each completed matrix.",
            ],
            "completed_matrices": sorted(results.keys()),
            "completed_count": len(results),
            "run_complete": bool(run_complete),
        },
        "results": results,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, path)


def _partial_out_path(out_path: Path) -> Path:
    if out_path.suffix == ".json":
        return out_path.with_name(out_path.stem + ".partial.json")
    return out_path.with_name(out_path.name + ".partial")


def _run_single_matrix(name: str, matrices_dir: str, args_dict: dict) -> tuple[str, dict, list[str]]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    matrices_root = Path(matrices_dir).expanduser().resolve()
    args = argparse.Namespace(**args_dict)
    log_lines = []

    A, b = load_problem({"name": name}, matrices_root)
    log_lines.append(f"n={A.shape[0]}, nnz={A.nnz}")

    specs = _method_specs(args)
    method_runs: dict[str, list[dict]] = {}
    for key in args.methods:
        spec = specs[key]
        runs: list[dict] = []
        if spec["type"] == "fixed":
            runs.append(
                run_fixed_gmres(
                    A,
                    b,
                    spec["restart"],
                    args.tolerance,
                    args.absolute_tolerance,
                    args.max_total_arnoldi,
                )
            )
        elif spec["type"] == "angle":
            runs.append(
                run_angle_gmres(
                    A,
                    b,
                    m_max=spec["m_max"],
                    m_min=spec["m_min"],
                    decrement=spec["decrement"],
                    tolerance=args.tolerance,
                    absolute_tolerance=args.absolute_tolerance,
                    max_total_arnoldi=args.max_total_arnoldi,
                    beta_small=args.angle_beta_small,
                    beta_large=args.angle_beta_large,
                )
            )
        elif spec["type"] == "rand":
            for offset in range(args.num_seeds):
                seed = args.base_seed + offset
                runs.append(
                    run_rand_gmres(
                        A,
                        b,
                        restart_values=spec["restart_values"],
                        tolerance=args.tolerance,
                        absolute_tolerance=args.absolute_tolerance,
                        max_total_arnoldi=args.max_total_arnoldi,
                        seed=seed,
                    )
                )
        elif spec["type"] == "dqn":
            for offset in range(args.num_seeds):
                seed = args.base_seed + offset
                runs.append(
                    run_dqn_gmres(
                        A,
                        b,
                        m_max=spec["m_max"],
                        history_length=args.dqn_history_length,
                        tolerance=args.tolerance,
                        absolute_tolerance=args.absolute_tolerance,
                        max_total_arnoldi=args.max_total_arnoldi,
                        gamma=args.dqn_gamma,
                        lambda_work=args.dqn_lambda_work,
                        convergence_bonus=args.dqn_convergence_bonus,
                        learning_rate=args.dqn_learning_rate,
                        buffer_size=args.dqn_buffer_size,
                        learning_starts=args.dqn_learning_starts,
                        batch_size=args.dqn_batch_size,
                        target_update_interval=args.dqn_target_update_interval,
                        exploration_fraction=args.dqn_exploration_fraction,
                        exploration_final_eps=args.dqn_exploration_final_eps,
                        device=args.device,
                        seed=seed,
                    )
                )
        else:
            raise ValueError(f"Unsupported method type: {spec['type']}")

        method_runs[key] = runs
        if spec["seeds"]:
            for offset, run in enumerate(runs):
                seed = args.base_seed + offset
                log_lines.append(
                    f"  seed={seed:02d}  {spec['label']} arnoldi={run['total_arnoldi']:7d} "
                    f"time={run['elapsed_seconds']:8.2f}s"
                )
        else:
            run = runs[0]
            log_lines.append(
                f"  {spec['label']} arnoldi={run['total_arnoldi']:7d} "
                f"time={run['elapsed_seconds']:8.2f}s"
            )

    row = {
        "shape": [int(A.shape[0]), int(A.shape[1])],
        "nnz": int(A.nnz),
        "methods": {
            key: {"runs": runs, "summary": summarise_runs(runs)}
            for key, runs in method_runs.items()
        },
    }
    return name, row, log_lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default="matrices/test")
    parser.add_argument("--matrix-names", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--max-total-arnoldi", type=int, default=100_000)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "gmres20",
            "gmres60",
            "rand_gmres_10_20",
            "rand_gmres_10_20_30_40_50_60",
            "gmres_rl_m20",
            "gmres_rl_m60",
            "angle_gmres_10_20",
            "angle_gmres_10_20_30_40_50_60",
        ],
    )

    parser.add_argument("--gmres20-restart", type=int, default=20)
    parser.add_argument("--gmres60-restart", type=int, default=60)
    parser.add_argument("--rand-restarts-short", nargs="+", type=int, default=[10, 20])
    parser.add_argument(
        "--rand-restarts-wide", nargs="+", type=int, default=[10, 20, 30, 40, 50, 60]
    )
    parser.add_argument("--angle-short-max", type=int, default=20)
    parser.add_argument("--angle-short-min", type=int, default=3)
    parser.add_argument("--angle-short-decrement", type=int, default=3)
    parser.add_argument("--angle-wide-max", type=int, default=60)
    parser.add_argument("--angle-wide-min", type=int, default=3)
    parser.add_argument("--angle-wide-decrement", type=int, default=3)
    parser.add_argument(
        "--angle-beta-small",
        type=float,
        default=float(math.cos(math.radians(80.0))),
    )
    parser.add_argument(
        "--angle-beta-large",
        type=float,
        default=float(math.cos(math.radians(8.0))),
    )

    parser.add_argument("--dqn-history-length", type=int, default=5)
    parser.add_argument("--dqn-gamma", type=float, default=0.925)
    parser.add_argument("--dqn-lambda-work", type=float, default=0.01)
    parser.add_argument("--dqn-convergence-bonus", type=float, default=10)
    parser.add_argument("--dqn-learning-rate", type=float, default=3e-3)
    parser.add_argument("--dqn-buffer-size", type=int, default=10_000)
    parser.add_argument("--dqn-learning-starts", type=int, default=25)
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-target-update-interval", type=int, default=100)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=0.10)
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=0.01)

    parser.add_argument(
        "--out",
        type=str,
        default="logs/peairs_style_159_suite.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    matrices_root = Path(args.matrices_dir).expanduser().resolve()
    out_path = Path(args.out)
    partial_path = _partial_out_path(out_path)
    names = (
        list(args.matrix_names)
        if args.matrix_names is not None
        else discover_matrix_names(matrices_root)
    )
    if args.limit is not None:
        names = names[: args.limit]
    if not names:
        raise ValueError("No matrices selected.")

    results = {}
    job_args = vars(args).copy()

    if args.n_jobs <= 1:
        for idx, name in enumerate(names, start=1):
            print(f"\n{'=' * 72}")
            print(f"[{idx}/{len(names)}] {name}")
            print(f"{'=' * 72}")
            matrix_name, row, log_lines = _run_single_matrix(
                name, str(matrices_root), job_args
            )
            for line in log_lines:
                print(line)
            results[matrix_name] = row
            _write_json_atomic(
                partial_path,
                _build_payload(
                    matrices_root=matrices_root,
                    names=names,
                    args=args,
                    results=results,
                    run_complete=False,
                ),
            )
    else:
        print(f"Running {len(names)} matrices with n_jobs={args.n_jobs}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            future_to_meta = {
                executor.submit(_run_single_matrix, name, str(matrices_root), job_args): (idx, name)
                for idx, name in enumerate(names, start=1)
            }
            for future in concurrent.futures.as_completed(future_to_meta):
                idx, name = future_to_meta[future]
                print(f"\n{'=' * 72}")
                print(f"[{idx}/{len(names)}] {name}")
                print(f"{'=' * 72}")
                matrix_name, row, log_lines = future.result()
                for line in log_lines:
                    print(line)
                results[matrix_name] = row
                _write_json_atomic(
                    partial_path,
                    _build_payload(
                        matrices_root=matrices_root,
                        names=names,
                        args=args,
                        results=results,
                        run_complete=False,
                    ),
                )

    payload = _build_payload(
        matrices_root=matrices_root,
        names=names,
        args=args,
        results=results,
        run_complete=True,
    )
    _write_json_atomic(out_path, payload)
    print(f"\nSaved benchmark log to {out_path}")


if __name__ == "__main__":
    main()
