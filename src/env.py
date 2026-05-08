"""
Discrete-action GMRES environment with the shaped reward. Does not include full residual vector.

Observation:
  - last k restart choices (normalized)
  - last k log relative residual norms

Action:
  - index into restart values 1..m_max
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GMRESEnv(gym.Env):
    """
    Gym environment for controlling GMRES(m) restart values on a fixed linear system Ax=b.

    The agent observes the last k residual norms and restart choices, but not the full
    residual vector, and selects a discrete restart value m in {1, ..., m_max}.
    GMRES(m) is run for one restart cycle, and the reward is [to be determined]
    """

    def __init__(
        self,
        A,
        b,
        m_max=20,
        tolerance=1e-5,
        absolute_tolerance=1e-12,
        max_cycles=200,
        history_length=5,
        lambda_work=0.0124,
        gamma_shape=0.9695,
        convergence_bonus=22.09,
    ):
        m_max = int(m_max)
        if m_max < 1:
            raise ValueError(f"m_max must be >= 1, got {m_max}")
        history_length = int(history_length)
        if history_length < 1:
            raise ValueError(
                f"history_length must be >= 1, got {history_length}"
            )

        super().__init__()

        # Linear system
        self.A = A
        self.b = b
        self.b_norm = max(float(np.linalg.norm(b)), 1e-12)
        self.n = A.shape[0]
        self.m_max = m_max
        self.tolerance = float(tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.max_cycles = int(max_cycles)
        self.history_length = history_length

        # Shaped-reward config
        self.lambda_work = float(lambda_work)
        self.gamma_shape = float(gamma_shape)
        self.convergence_bonus = float(convergence_bonus)

        # History-state config
        self.action_ms = tuple(range(1, m_max + 1))
        self.total_arnoldi = 0

        # discrete action over restart values 1..m_max.
        self.action_space = spaces.Discrete(len(self.action_ms))

        # Observation:
        #   [last k restart choices / m_max, last k log relative residual norms]
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    np.zeros(self.history_length, dtype=np.float32),
                    np.full(self.history_length, np.finfo(np.float32).min, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.ones(self.history_length, dtype=np.float32),
                    np.full(self.history_length, np.finfo(np.float32).max, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

        # Runtime state
        self.x = None
        self.current_residual_vector = None
        self.current_residual_norm = None
        self.cycle_count = 0
        self.restart_history = None
        self.residual_history = None

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.x = np.zeros(self.n)
        self.current_residual_vector = self.b - self.A @ self.x
        self.current_residual_norm = float(np.linalg.norm(self.current_residual_vector))
        self.cycle_count = 0
        self.total_arnoldi = 0
        self.restart_history = np.zeros(self.history_length, dtype=np.float32)
        self.residual_history = np.zeros(self.history_length, dtype=np.float32)
        self.residual_history[-1] = np.float32(
            np.log(max(self._relative_residual(self.current_residual_norm), 1e-12))
        )

        obs = self._build_observation()
        info = {
            "residual_norm": self.current_residual_norm,
            "relative_residual_norm": self._relative_residual(self.current_residual_norm),
            "cycle_count": self.cycle_count,
            "total_arnoldi": self.total_arnoldi,
        }
        return obs, info

    def step(self, action):
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.action_ms):
            raise ValueError(f"action index {action_idx} outside valid range")

        m = int(self.action_ms[action_idx])
        prev_norm = self.current_residual_norm

        # Early-convergence guard: if we're already at tolerance, terminate cleanly.
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

        # Run one GMRES(m) restart cycle.
        self.x, self.current_residual_vector, actual_m = self._execute_gmres_cycle(
            self.x, self.current_residual_vector, m
        )
        self.current_residual_norm = float(np.linalg.norm(self.current_residual_vector))
        self.cycle_count += 1
        self.total_arnoldi += actual_m

        # Reward
        curr_rel = self._relative_residual(self.current_residual_norm)
        reward = self._compute_reward(
            prev_norm, prev_rel, self.current_residual_norm, curr_rel, m
        )

        # Update k-step history state.
        self.restart_history = np.roll(self.restart_history, -1)
        self.restart_history[-1] = np.float32(m / self.m_max)
        self.residual_history = np.roll(self.residual_history, -1)
        self.residual_history[-1] = np.float32(np.log(max(curr_rel, 1e-12)))

        # Termination / truncation
        terminated = bool(self._is_converged(self.current_residual_norm, curr_rel))
        truncated = bool(self.cycle_count >= self.max_cycles) and not terminated

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

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def _compute_reward(self, prev_norm, prev_rel, curr_norm, curr_rel, m):
        """
        Potential-based shaped reward (Ng, Harada, Russell 1999) with potential

            Phi_tau(s) = -log( max(rho, tau) / tau )   <=  0,

        where rho = ||r|| / ||b|| is the relative residual and tau is the
        convergence tolerance on rho. Phi is non-positive everywhere and
        equals 0 exactly at convergence (rho <= tau).

        The shaping term is gamma * Phi(s_{t+1}) - Phi(s_t), so

            R_t = -lambda_work * m_t
                  + log( max(rho_t,     tau) / tau )
                  - gamma_shape * log( max(rho_{t+1}, tau) / tau )
                  + convergence_bonus * 1{rho_{t+1} < tau <= rho_t}.

        Because this is a true PBRS term with the same gamma the agent uses
        for value bootstrapping, it preserves the optimal policy.
        """
        tau = self.tolerance
        log_prev = np.log(max(prev_rel, tau) / tau)
        log_curr = np.log(max(curr_rel, tau) / tau)
        reward = -self.lambda_work * m + log_prev - self.gamma_shape * log_curr
        if self._is_converged(curr_norm, curr_rel) and not self._is_converged(prev_norm, prev_rel):
            reward += self.convergence_bonus
        return reward

    def _relative_residual(self, residual_norm):
        return float(residual_norm) / self.b_norm

    def _is_converged(self, residual_norm, relative_residual):
        return bool(
            float(relative_residual) < self.tolerance
            or float(residual_norm) < self.absolute_tolerance
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _build_observation(self):
        """Return the rolling k-step history observation."""
        return np.concatenate(
            [self.restart_history, self.residual_history],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # GMRES(m) core
    # ------------------------------------------------------------------ #

    def _execute_gmres_cycle(self, x_current, r_current, m):
        """
        Run one restart cycle of GMRES(m) starting from x_current with residual
        r_current = b - A x_current. Returns updated
        `(x_next, r_next, actual_m)`, where `actual_m` is the number of Arnoldi
        steps performed before either reaching `m` or breaking down early.
        """
        V, H, beta, actual_m, breakdown = self._arnoldi_iteration(
            self.A, r_current, m, tol=1e-14
        )

        # If the initial residual was essentially zero, Arnoldi signals via
        # actual_m == 0. Nothing to do.
        if actual_m == 0:
            return x_current, r_current, 0

        # Solve min_y || beta * e_1 - H y ||_2.
        e1 = np.zeros(actual_m + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H, e1, rcond=None)

        # Update iterate using the Arnoldi basis.
        x_next = x_current + V[:, :actual_m] @ y

        # Recompute residual explicitly for numerical accuracy.
        r_next = self.b - self.A @ x_next
        return x_next, r_next, actual_m

    @staticmethod
    def _arnoldi_iteration(A, r, m, tol=1e-14):
        """
        Perform up to m steps of Arnoldi iteration starting from r.

        Returns
        -------
        V : (n, actual_m + 1) orthonormal basis
        H : (actual_m + 1, actual_m) upper Hessenberg
        beta : float, ||r||
        actual_m : int, number of steps completed
        breakdown : bool, True if early termination due to near-zero direction
        """
        n = r.shape[0]
        beta = float(np.linalg.norm(r))

        # If residual is already tiny, return empty Arnoldi output rather than raise.
        if beta < tol:
            V = np.zeros((n, 1))
            H = np.zeros((1, 0))
            return V, H, beta, 0, True

        V = np.zeros((n, m + 1), dtype=float)
        H = np.zeros((m + 1, m), dtype=float)
        V[:, 0] = r / beta

        actual_m = 0
        breakdown = False

        for j in range(m):
            vj = V[:, j]
            w = A(vj) if callable(A) else A @ vj

            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = float(np.linalg.norm(w))
            actual_m = j + 1

            if H[j + 1, j] < tol:
                breakdown = True
                break

            V[:, j + 1] = w / H[j + 1, j]

        V = V[:, : actual_m + 1]
        H = H[: actual_m + 1, :actual_m]
        return V, H, beta, actual_m, breakdown
