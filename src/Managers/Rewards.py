from __future__ import annotations
import json
import torch
from datetime import datetime, timezone
from pathlib import Path
from quadcopter_lift_env_cfg import NUM_DRONES

# ---- NaN debug: log only the very first occurrence per run ----
_nan_logged: bool = False
_NAN_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "nan_debug.jsonl"


def _log_nan_event(source: str, tensor_name: str, tensor: torch.Tensor, step: int) -> None:
    """Append a single JSON record to nan_debug.jsonl and print to stdout.
    Called at most once per Python session (guarded by module-level flag)."""
    global _nan_logged
    if _nan_logged:
        return
    _nan_logged = True

    has_nan = bool(torch.isnan(tensor).any())
    has_inf = bool(torch.isinf(tensor).any())
    n_affected = int((torch.isnan(tensor) | torch.isinf(tensor)).any(dim=-1).sum())

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,          # "rewards" | "observations"
        "tensor": tensor_name,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "n_envs_affected": n_affected,
        "step": step,
        "tensor_shape": list(tensor.shape),
    }

    _NAN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _NAN_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    print(
        f"[NaN DEBUG] first occurrence | source={source} tensor={tensor_name} "
        f"nan={has_nan} inf={has_inf} envs_affected={n_affected} step={step} "
        f"→ {_NAN_LOG_PATH}"
    )


# ---------------------------------------------------------------------------
# Weights and scales — move to config later
# ---------------------------------------------------------------------------
# Task
W_GOAL_HEIGHT    =  3.0    # increase — primary learning signal, needs to dominate

W_GOAL_DIST      =  2.0    # increase slightly — complements height
W_GOAL_DIST_POTENTIAL = 1.0 # new shaping term to encourage progress toward goal

W_GROUND_CONTACT = -1.5    # enough to break do-nothing without overwhelming everything

# Crate stability
W_BALANCE        =  0.5    # low early — don't reward stillness over lifting
W_TWIST          = -0.1    # very low — policy needs freedom to yaw-correct early on

# Drone behaviour
W_SMOOTH_ACTION  = -0.01   # halved — don't over-constrain exploration early

W_PROXIMITY      = -0.05   # new reward to encourage drones to keep distance
W_PROXIMITY_SIGMA = 0.1     # distance scale for proximity penalty
W_FORMATION      =  0.1    # penalise drones drifting from their reference corners

# Terminal
W_SUCCESS        = 10.0    # keep — strong goal signal
W_CRASH          = -3.0    # reduced — too harsh early discourages exploration

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
        self.curriculum = None   # set by env after CurriculumManager is created
        self.dt = env.physics_dt
        # Precompute pair indices for proximity (same as termination manager)
        pairs = [(i, j) for i in range(NUM_DRONES) for j in range(i + 1, NUM_DRONES)]
        self._pair_idx = torch.tensor(pairs, device=self.device)   # (6, 2)

        # Previous action state for smoothness reward — (n, A, 4)
        self._prev_action_state = torch.zeros(
            env.num_envs, NUM_DRONES, 4, device=self.device
        )
        self._prev_dist = torch.zeros(env.num_envs, device=self.device)  # for potential-based shaping

        # Episode accumulator for logging
        self._episode_sums: dict[str, torch.Tensor] = {
            k: torch.zeros(env.num_envs, device=self.device)
            for k in [
                "goal_height", "goal_dist", 
                "goal_dist_potential",
                "balance", "twist", "ground_contact",
                "smooth_action", "proximity",
                "formation_deviation",
                "success", "crash",
            ]
        }

        print("[RewardManager] initialised")
        print(f"  W_GOAL_HEIGHT={W_GOAL_HEIGHT}  W_GOAL_DIST={W_GOAL_DIST}")
        print(f"  W_BALANCE={W_BALANCE}  W_TWIST={W_TWIST}")
        print(f"  W_SMOOTH_ACTION={W_SMOOTH_ACTION}  W_GROUND_CONTACT={W_GROUND_CONTACT}")
        print(f"  W_FORMATION={W_FORMATION}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset internal buffers and return the average episode rewards for the resetting envs."""
        self._prev_action_state[env_ids] = 0.0
        # crate_pos = self.env._crate.data.root_pos_w[env_ids, :3]
        # self._prev_dist[env_ids] = (
        #     self.env._goal_pos_w[env_ids] - crate_pos
        # ).norm(dim=-1)
        
        # Prepare a dictionary to hold the final logged values for these environments
        logged_rewards = {}
        
        if len(env_ids) > 0:
            for key, val_tensor in self._episode_sums.items():
                # Average the accumulated reward across the resetting environments
                avg_sum = torch.mean(val_tensor[env_ids]).item()
                
                # Normalize by episode length (optional, but standard in IsaacLab)
                avg_reward_per_step = avg_sum / self.env.max_episode_length
                
                # Format key with 'rew_' prefix for RSL-RL
                logged_rewards[f"rew_{key}"] = avg_reward_per_step
                
                # Clear the buffer
                self._episode_sums[key][env_ids] = 0.0
                
        return logged_rewards
    def reset_dist(self, env_ids):
        crate_pos = self.env._crate.data.root_pos_w[env_ids, :3]
        self._prev_dist[env_ids] = (
            self.env._goal_pos_w[env_ids] - crate_pos
        ).norm(dim=-1)

    def compute(self, terminated: torch.Tensor, timed_out: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns shared reward broadcast to all agents.
        {"drone_i": (num_envs,)}
        """
        dt = self.env.step_dt

        # Curriculum factors (0→1) for gated rewards
        cx_goal_dist  = self.curriculum.get_factor("goal_dist")           if self.curriculum else 1.0
        cx_proximity  = self.curriculum.get_factor("proximity")           if self.curriculum else 1.0
        cx_formation  = self.curriculum.get_factor("formation_deviation") if self.curriculum else 1.0

        # 1. Compute raw, UNSCALED components for logging
        raw_components = {
            "goal_height"    : W_GOAL_HEIGHT     * self._rew_goal_height(),
            "goal_dist"      : W_GOAL_DIST       * cx_goal_dist  * self._rew_goal_dist(),
            # "goal_vel_align" : W_GOAL_VEL_ALIGN  * self._rew_goal_vel_align(),
            "ground_contact" : W_GROUND_CONTACT  * self._rew_ground_contact(),
            "balance"        : W_BALANCE         * self._rew_balance(),
            "goal_dist_potential": W_GOAL_DIST_POTENTIAL      * cx_goal_dist  * self._rew_goal_dist_potential(),
            # "twist"          : W_TWIST           * self._rew_twist(),
            # "smooth_action"  : W_SMOOTH_ACTION   * self._rew_smooth_action(),
            "proximity"      : W_PROXIMITY       * cx_proximity  * self._rew_proximity(),
            "formation_deviation": W_FORMATION    * cx_formation  * self._rew_formation_deviation(),

        }

        # 2. Accumulate the raw values for TensorBoard logging
        for k, v in raw_components.items():
            self._episode_sums[k] += v

        
        total_reward = torch.zeros(self.env.num_envs, device=self.device)
        
        for k, v in raw_components.items():
            total_reward += v * dt   # Scale continuous rewards

        # ---- NaN/Inf detection — log first occurrence ----
        if not _nan_logged:
            step = int(self.env.episode_length_buf.max().item())
            for _tname, _t in [
                ("crate_pos",    self.env._crate.data.root_pos_w),
                ("drone_0_pos",  self.env._drones["drone_0"].data.root_pos_w),
                ("crate_ang_vel",self.env._crate.data.root_ang_vel_w),
            ]:
                if torch.isnan(_t).any() or torch.isinf(_t).any():
                    _log_nan_event("rewards", _tname, _t, step)
                    break  # log once; outer guard prevents further calls

        # Guard against NaN/Inf from physics instability — prevents
        # corrupting the policy network (which causes the
        # "normal expects all elements of std >= 0.0" crash).
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)
        total_reward = total_reward.clamp(-10.0, 10.0)

        return {name: total_reward for name in self.env.cfg.possible_agents}


    def get_episode_sums(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._episode_sums.items()}

    # ------------------------------------------------------------------
    # Reward terms — each returns (num_envs,)
    # ------------------------------------------------------------------

    def _rew_goal_height(self):
        z_crate  = self.env._crate.data.root_pos_w[:, 2]   # world Z
        z_goal   = self.env._goal_pos_w[:, 2]               # world Z of sampled goal
        height_error = (z_goal - z_crate).clamp(min=0.0)   # only penalise being below goal
        return torch.exp(-2.0 * height_error)
    
    def _rew_goal_dist(self):
        goal_pos  = self.env._goal_pos_w
        crate_pos = self.env._crate.data.root_pos_w[:, :3]
        dist = torch.norm(goal_pos - crate_pos, dim=-1)      # (n,)  NOT squared
        return torch.exp(-5.0 * dist)       
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

        # Crate angular velocity — roll (x) and pitch (y) only, not yaw
        omega  = self.env._crate.data.root_ang_vel_w                # (n, 3)
        # omega_xy_sq = (omega[:, 0] ** 2 + omega[:, 1] ** 2)        # (n,)  ||ω_xy||²

        omega_z_sq = omega[:, 2] ** 2      
        omega_xy_sq = (omega[:, 0] ** 2 + omega[:, 1] ** 2)        # (n,)  ||ω_xy||²

        term1 = torch.exp(-2.5 * omega_xy_sq)                  # (n,)
        term2 = torch.exp(-2.0 * omega_z_sq)                 # (n,)

        return  term1 + term2                                        # (n,)  ∈ (0, 2]

    def _rew_twist(self) -> torch.Tensor:
        """
        From paper:
            -0.6 * cx * ||θ_xy||

        θ_xy = roll + pitch angle of crate extracted from quaternion.
        Note: W_TWIST is negative, so this penalises tilt.
        We return the raw ||θ_xy|| here; sign comes from W_TWIST.
        """

        q = self.env._crate.data.root_quat_w    # (n, 4)  [w, x, y, z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Roll (rotation around X)
        roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))     # (n,)
        # Pitch (rotation around Y)
        pitch = torch.asin((2*(w*y - z*x)).clamp(-1.0, 1.0))       # (n,)

        theta_xy_norm = (roll.abs() + pitch.abs())                  # (n,)  ||θ_xy||

        return  theta_xy_norm                                   # (n,)

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
        return mse

    def _rew_proximity(self) -> torch.Tensor:
        pos = torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_pos_w
            for i in range(NUM_DRONES)],
            dim=1,
        )  # (n, A, 3)

        idx_a = self._pair_idx[:, 0]
        idx_b = self._pair_idx[:, 1]

        dist = (pos[:, idx_a] - pos[:, idx_b]).norm(dim=-1)  # (n, 6)

        # exp decays naturally — no hard threshold needed
        # at dist=0.15m (collision_radius): exp(-0.15/0.1) ≈ 0.22
        # at dist=0.5m (well separated):   exp(-0.5/0.1)  ≈ 0.007
        penalty = torch.exp(-dist / W_PROXIMITY_SIGMA)         # (n, 6)

        return penalty.sum(dim=-1)                           # (n,)

    def _rew_formation_deviation(self) -> torch.Tensor:
        crate_pos  = self.env._crate.data.root_pos_w          # (n, 3)
        drone_pos  = torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_pos_w for i in range(NUM_DRONES)],
            dim=1,
        )  # (n, A, 3)

        # How far is each drone from directly above the crate centre (XY only)
        delta_xy = drone_pos[:, :, :2] - crate_pos[:, :2].unsqueeze(1)  # (n, A, 2)
        lateral_pull = (delta_xy ** 2).sum(dim=-1)                       # (n, A)

        return -lateral_pull.mean(dim=-1)                                # (n,)
    def _rew_ground_contact(self) -> torch.Tensor:
        ground_force = self.env._crate_contact.data.net_forces_w[:, 0, :]  # (n, 3)
        in_contact   = ground_force.norm(dim=-1) > 0.5                      # (n,) bool
        return in_contact.float()   # 1.0 when crate touching ground, 0.0 when lifted
    def _rew_goal_dist_potential(self) -> torch.Tensor:
        """
        Potential-based shaping: reward = φ(s') - φ(s)
        φ(s) = -dist(crate, goal)
        Positive when crate moves closer, negative when it drifts away.
        Naturally small (delta distance per step) so won't overpower other terms.
        """
        goal_pos  = self.env._goal_pos_w
        crate_pos = self.env._crate.data.root_pos_w[:, :3]
        dist      = (goal_pos - crate_pos).norm(dim=-1)   # (n,)

        reward = self._prev_dist - dist                   # positive = got closer
        self._prev_dist = dist.detach().clone()
        #clamp reward to a reasonable range to prevent instability
        reward = reward.clamp(min=-0.1, max=0.1)
        return reward