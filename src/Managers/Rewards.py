from __future__ import annotations
import torch
from quadcopter_lift_env_cfg import NUM_DRONES


# ---------------------------------------------------------------------------
# Weights and scales — move to config later
# ---------------------------------------------------------------------------
# Task
W_GOAL_HEIGHT    =  3.0    # increase — primary learning signal, needs to dominate
W_GOAL_DIST      =  2.0    # increase slightly — complements height
W_GOAL_VEL_ALIGN =  0.5    # keep low — useful but secondary
W_GROUND_CONTACT = -1.5    # enough to break do-nothing without overwhelming everything

# Crate stability
W_BALANCE        =  0.3    # low early — don't reward stillness over lifting
W_TWIST          = -0.1    # very low — policy needs freedom to yaw-correct early on

# Drone behaviour
W_ALIVE          =  0.0    # removed as discussed
W_SMOOTH_ACTION  = -0.01   # halved — don't over-constrain exploration early
W_PROXIMITY      = -0.05   # keep — drones crashing into each other is always bad

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
        self.dt = env.physics_dt
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
                "balance", "twist", "ground_contact",
                "alive", "smooth_action", "proximity",
                "success", "crash",
            ]
        }

        print("[RewardManager] initialised")
        print(f"  W_GOAL_HEIGHT={W_GOAL_HEIGHT}  W_GOAL_DIST={W_GOAL_DIST}  "
              f"W_GOAL_VEL_ALIGN={W_GOAL_VEL_ALIGN}")
        print(f"  W_BALANCE={W_BALANCE}  W_TWIST={W_TWIST}")
        print(f"  W_ALIVE={W_ALIVE}  W_SMOOTH={W_SMOOTH_ACTION}  W_PROXIMITY={W_PROXIMITY}  W_GROUND_CONTACT={W_GROUND_CONTACT}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset internal buffers and return the average episode rewards for the resetting envs."""
        self._prev_action_state[env_ids] = 0.0
        
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

    def compute(self, terminated: torch.Tensor, timed_out: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns shared reward broadcast to all agents.
        {"drone_i": (num_envs,)}
        """
        dt = self.env.step_dt

        # 1. Compute raw, UNSCALED components for logging
        raw_components = {
            "goal_height"    : W_GOAL_HEIGHT     * self._rew_goal_height(),
            "goal_dist"      : W_GOAL_DIST       * self._rew_goal_dist(),
            # "goal_vel_align" : W_GOAL_VEL_ALIGN  * self._rew_goal_vel_align(),
            "ground_contact" : W_GROUND_CONTACT  * self._rew_ground_contact(),
            "balance"        : W_BALANCE         * self._rew_balance(),
            # "twist"          : W_TWIST           * self._rew_twist(),
            # "alive"          : W_ALIVE           * self._rew_alive(),
            # "smooth_action"  : W_SMOOTH_ACTION   * self._rew_smooth_action(),
            # "proximity"      : W_PROXIMITY       * self._rew_proximity(),
            
            # Terminal rewards (always unscaled)
            # "success"        : W_SUCCESS         * terminated.float() *
            #                    (self.env._crate.data.root_pos_w[:, 2] >=
            #                     self.env.cfg.goal_pos[2] - 0.1).float(),
            # "crash"          : W_CRASH           * terminated.float() *
            #                    (self.env._crate.data.root_pos_w[:, 2] <
            #                     self.env.cfg.goal_pos[2] - 0.1).float(),
        }

        # 2. Accumulate the raw values for TensorBoard logging
        for k, v in raw_components.items():
            self._episode_sums[k] += v

        # 3. Apply dt scaling ONLY for the RL algorithm's training signal
        continuous_keys = [
            "goal_height", "goal_dist", "goal_vel_align", "ground_contact", 
            "balance", "twist", "smooth_action", "proximity"
            # add "alive" here if you uncomment it
        ]
        
        total_reward = torch.zeros(self.env.num_envs, device=self.device)
        
        for k, v in raw_components.items():
            if k in continuous_keys:
                total_reward += v * dt   # Scale continuous rewards
            else:
                total_reward += v        # Add terminal events as-is (success, crash)

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
        return torch.exp(-1.0 * dist)       
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

        omega_z_sq = omega[:, 2] ** 2                                   # (n,)  ω_z²

        # term1 = torch.exp(-2.5 * cx * (v_z ** 2))                  # (n,)
        term2 = torch.exp(-2.0 * omega_z_sq)                 # (n,)

        return  term2                                        # (n,)  ∈ (0, 2]

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
        return mse

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

    def _rew_ground_contact(self) -> torch.Tensor:
        ground_force = self.env._crate_contact.data.net_forces_w[:, 0, :]  # (n, 3)
        in_contact   = ground_force.norm(dim=-1) > 0.5                      # (n,) bool
        return in_contact.float()   # 1.0 when crate touching ground, 0.0 when lifted
