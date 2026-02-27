from __future__ import annotations
import torch
from quadcopter_lift_env_cfg import NUM_DRONES


# ---------------------------------------------------------------------------
# Weights and scales — move to config later
# ---------------------------------------------------------------------------

# Task
W_GOAL_HEIGHT        =  2.0    # exponential reward for crate Z
W_GOAL_DIST          =  1.5    # tanh-shaped distance to goal
W_GOAL_VEL_ALIGN     =  1.0    # reward velocity aimed at goal

# Crate stability (from paper)
W_BALANCE            =  1.3    # balance: exp terms on crate angular vel + tilt
W_TWIST              = -0.6    # twist: penalise crate yaw rate

# Drone behaviour
W_ALIVE              =  0.5    # per-step survival bonus
W_SMOOTH_ACTION      =  0.02   # small bonus for smooth action (positive)
W_PROXIMITY          = -0.05   # small penalty for drones getting close

# Terminal
W_SUCCESS            = 10.0    # bonus on reaching goal
W_CRASH              = -5.0    # penalty on termination

# Scaling / shaping constants
GOAL_HEIGHT_SIGMA    = 0.5     # exp(-z_err² / sigma) sharpness
BALANCE_CX           = 1.0     # c_x in balance formula from paper
PROXIMITY_SIGMA      = 0.3     # soft proximity falloff (metres)
PROXIMITY_MIN_DIST   = 0.25    # start penalising below this distance (metres)
SMOOTH_ACTION_SIGMA  = 0.1     # action delta scale for smoothness bonus


# ---------------------------------------------------------------------------
# Reward Manager
# ---------------------------------------------------------------------------

class RewardManager:
    """
    Reward manager for cooperative crate lift.

    All rewards are computed over (num_envs,) and summed into a single
    scalar per env, then broadcast to all agents (shared team reward).

    Formula references:
        balance: 1.3 * (exp(-2.5*cx * |v_z|²) + exp(-2*cx * ||ω_xy||²))
        twist:  -0.6 * cx * ||θ_xy||
    where cx = BALANCE_CX, v_z = crate vertical vel, ω_xy = crate roll/pitch rate,
    θ_xy = crate roll/pitch angle extracted from quaternion.
    """

    def __init__(self, env):
        self.env    = env
        self.device = env.device

        # Precompute pair indices for proximity (same as termination manager)
        pairs = [(i, j) for i in range(NUM_DRONES) for j in range(i + 1, NUM_DRONES)]
        self._pair_idx = torch.tensor(pairs, device=self.device)   # (6, 2)

        # Previous action state for smoothness reward — (n, A, 4)
        self._prev_action_state = torch.zeros(
            env.num_envs, NUM_DRONES, 4, device=self.device
        )

        # Episode accumulator for logging
        self._episode_sums: dict[str, torch.Tensor] = {
            k: torch.zeros(env.num_envs, device=self.device)
            for k in [
                "goal_height", "goal_dist", "goal_vel_align",
                "balance", "twist",
                "alive", "smooth_action", "proximity",
                "success", "crash",
            ]
        }

        print("[RewardManager] initialised")
        print(f"  W_GOAL_HEIGHT={W_GOAL_HEIGHT}  W_GOAL_DIST={W_GOAL_DIST}  "
              f"W_GOAL_VEL_ALIGN={W_GOAL_VEL_ALIGN}")
        print(f"  W_BALANCE={W_BALANCE}  W_TWIST={W_TWIST}")
        print(f"  W_ALIVE={W_ALIVE}  W_SMOOTH={W_SMOOTH_ACTION}  W_PROXIMITY={W_PROXIMITY}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_action_state[env_ids] = 0.0
        for k in self._episode_sums:
            self._episode_sums[k][env_ids] = 0.0

    def compute(self, terminated: torch.Tensor, timed_out: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns shared reward broadcast to all agents.
        {"drone_i": (num_envs,)}
        """
        components = {
            "goal_height"    : W_GOAL_HEIGHT     * self._rew_goal_height(),
            "goal_dist"      : W_GOAL_DIST       * self._rew_goal_dist(),
            "goal_vel_align" : W_GOAL_VEL_ALIGN  * self._rew_goal_vel_align(),
            "balance"        : W_BALANCE         * self._rew_balance(),
            "twist"          : W_TWIST           * self._rew_twist(),
            "alive"          : W_ALIVE           * self._rew_alive(),
            "smooth_action"  : W_SMOOTH_ACTION   * self._rew_smooth_action(),
            "proximity"      : W_PROXIMITY       * self._rew_proximity(),
            "success"        : W_SUCCESS         * terminated.float() *
                               (self.env._crate.data.root_pos_w[:, 2] >=
                                self.env.cfg.goal_pos[2] - 0.1).float(),
            "crash"          : W_CRASH           * terminated.float() *
                               (self.env._crate.data.root_pos_w[:, 2] <
                                self.env.cfg.goal_pos[2] - 0.1).float(),
        }

        # Accumulate for logging
        for k, v in components.items():
            self._episode_sums[k] += v

        total = sum(components.values())   # (n,)

        return {name: total for name in self.env.cfg.possible_agents}

    def get_episode_sums(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._episode_sums.items()}

    # ------------------------------------------------------------------
    # Reward terms — each returns (num_envs,)
    # ------------------------------------------------------------------

    def _rew_goal_height(self) -> torch.Tensor:
        """
        Exponential reward for crate reaching target height.
        r = exp(-(z_goal - z_crate)² / sigma)
        Peaks at 1.0 when crate is exactly at goal height.
        """
        z_crate  = self.env._crate.data.root_pos_w[:, 2]          # (n,)
        z_goal   = self.env._goal_pos_w[:, 2]                      # (n,)
        z_err    = z_goal - z_crate                                # (n,)
        return torch.exp(-(z_err ** 2) / GOAL_HEIGHT_SIGMA)        # (n,)  ∈ (0, 1]

    def _rew_goal_dist(self) -> torch.Tensor:
        """
        Tanh-shaped 3D distance reward.
        r = 1 - tanh(‖goal - crate‖ / 0.5)   ∈ (0, 1]
        """
        dist = torch.norm(
            self.env._goal_pos_w - self.env._crate.data.root_pos_w, dim=-1
        )   # (n,)
        return 1.0 - torch.tanh(dist / 0.5)

    def _rew_goal_vel_align(self) -> torch.Tensor:
        """
        Reward crate velocity that is aligned toward the goal.
        r = dot(crate_vel, goal_dir) — positive when moving toward goal.
        Clipped to [0, ∞) so moving away gives zero, not negative.
        """
        to_goal  = self.env._goal_pos_w - self.env._crate.data.root_pos_w   # (n, 3)
        dist     = to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir = to_goal / dist                                            # (n, 3) unit

        crate_vel = self.env._crate.data.root_lin_vel_w                     # (n, 3)
        alignment = (crate_vel * goal_dir).sum(dim=-1)                      # (n,) dot product

        return alignment.clamp(min=0.0)   # only reward progress, not regression

    def _rew_balance(self) -> torch.Tensor:
        """
        From paper:
            1.3 * (exp(-2.5*cx * |v_z|²) + exp(-2*cx * ||ω_xy||²))

        v_z   = crate vertical velocity
        ω_xy  = crate roll + pitch angular velocity (not yaw)

        Both terms peak at 1.0 when the crate is perfectly stable.
        Note: W_BALANCE is applied externally in compute(), so we return
        the raw sum here (max value = 2.0).
        """
        cx = BALANCE_CX

        # Crate vertical velocity
        v_z   = self.env._crate.data.root_lin_vel_w[:, 2]          # (n,)

        # Crate angular velocity — roll (x) and pitch (y) only, not yaw
        omega  = self.env._crate.data.root_ang_vel_w                # (n, 3)
        omega_xy_sq = (omega[:, 0] ** 2 + omega[:, 1] ** 2)        # (n,)  ||ω_xy||²

        term1 = torch.exp(-2.5 * cx * (v_z ** 2))                  # (n,)
        term2 = torch.exp(-2.0 * cx * omega_xy_sq)                 # (n,)

        return term1 + term2                                        # (n,)  ∈ (0, 2]

    def _rew_twist(self) -> torch.Tensor:
        """
        From paper:
            -0.6 * cx * ||θ_xy||

        θ_xy = roll + pitch angle of crate extracted from quaternion.
        Note: W_TWIST is negative, so this penalises tilt.
        We return the raw ||θ_xy|| here; sign comes from W_TWIST.
        """
        cx = BALANCE_CX

        q = self.env._crate.data.root_quat_w    # (n, 4)  [w, x, y, z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Roll (rotation around X)
        roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))     # (n,)
        # Pitch (rotation around Y)
        pitch = torch.asin((2*(w*y - z*x)).clamp(-1.0, 1.0))       # (n,)

        theta_xy_norm = (roll.abs() + pitch.abs())                  # (n,)  ||θ_xy||

        return cx * theta_xy_norm                                   # (n,)

    def _rew_alive(self) -> torch.Tensor:
        """Constant per-step survival bonus. (n,)"""
        return torch.ones(self.env.num_envs, device=self.device)

    def _rew_smooth_action(self) -> torch.Tensor:
        """
        Small bonus for smooth actions.
        r = exp(-||action_t - action_{t-1}||² / sigma)
        Peaks at 1.0 when action doesn't change at all.
        Returns mean over all drones.
        """
        current = self.env._action_manager.get_state()              # (n, A, 4)
        delta   = (current - self._prev_action_state) ** 2         # (n, A, 4)
        mse     = delta.mean(dim=-1).mean(dim=-1)                   # (n,)

        self._prev_action_state = current.detach().clone()

        return torch.exp(-mse / SMOOTH_ACTION_SIGMA)                # (n,)  ∈ (0, 1]

    def _rew_proximity(self) -> torch.Tensor:
        """
        Soft continuous proximity penalty for pairs of drones.
        r = sum over pairs of exp(-dist / sigma) when dist < min_dist, else 0.
        Note: W_PROXIMITY is negative, so this penalises closeness.
        """
        pos = torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_pos_w
             for i in range(NUM_DRONES)],
            dim=1,
        )   # (n, A, 3)

        idx_a = self._pair_idx[:, 0]
        idx_b = self._pair_idx[:, 1]

        dist = (pos[:, idx_a] - pos[:, idx_b]).norm(dim=-1)        # (n, 6)

        # Only activate when closer than threshold
        active  = (dist < PROXIMITY_MIN_DIST).float()              # (n, 6)
        penalty = torch.exp(-dist / PROXIMITY_SIGMA) * active      # (n, 6)

        return penalty.sum(dim=-1)                                  # (n,)
