# marl_wrapper.py

from __future__ import annotations
import torch
from quadcopter_lift_env_cfg import NUM_DRONES


class FlattenedMARLWrapper:
    """
    Flattens a CoopLiftEnv from MARL shape to single-agent shape so
    RSL-RL can treat every drone as an independent environment instance.

    CoopLiftEnv shape          →  RSL-RL shape
    ─────────────────────────────────────────────
    obs:     (n, A, obs_dim)   →  (n*A, obs_dim)
    actions: (n*A, act_dim)    →  split back to (n, A, act_dim)
    rewards: (n, A)            →  (n*A,)
    dones:   (n, A)            →  (n*A,)
    """

    def __init__(self, env):
        self.env       = env
        self.num_envs  = env.num_envs * NUM_DRONES   # RSL-RL sees this
        self.device    = env.device

        # Single agent obs/action dims
        self.obs_dim    = env._obs_manager.obs_dim
        self.action_dim = env._action_manager.action_dim

        # RSL-RL expects these attributes
        self.num_obs     = self.obs_dim
        self.num_actions = self.action_dim

    # ------------------------------------------------------------------
    # Core interface RSL-RL calls
    # ------------------------------------------------------------------

    def reset(self):
        obs_dict, info = self.env.reset()
        return self._flatten_obs(obs_dict), info

    def step(self, actions: torch.Tensor):
        """
        Args:
            actions: (n*A, act_dim) — RSL-RL's flat action tensor

        Returns:
            obs:     (n*A, obs_dim)
            rewards: (n*A,)
            dones:   (n*A,)
            info:    dict
        """
        action_dict = self._unflatten_actions(actions)

        obs_dict, reward_dict, terminated_dict, truncated_dict, info = \
            self.env.step(action_dict)

        obs     = self._flatten_obs(obs_dict)           # (n*A, obs_dim)
        rewards = self._flatten_rewards(reward_dict)    # (n*A,)
        dones   = self._flatten_dones(
            terminated_dict, truncated_dict
        )                                               # (n*A,)

        return obs, rewards, dones, info

    # ------------------------------------------------------------------
    # Flatten / unflatten helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        {"drone_i": (n, obs_dim)} → stack → (n, A, obs_dim) → (n*A, obs_dim)
        Order: [env_0_drone_0, env_0_drone_1, ..., env_n_drone_A]
        """
        stacked = torch.stack(
            [obs_dict[f"drone_{i}"] for i in range(NUM_DRONES)],
            dim=1,
        )   # (n, A, obs_dim)
        return stacked.reshape(self.num_envs, self.obs_dim)   # (n*A, obs_dim)

    def _unflatten_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        (n*A, act_dim) → (n, A, act_dim) → {"drone_i": (n, act_dim)}
        """
        n = self.env.num_envs
        unflat = actions.reshape(n, NUM_DRONES, self.action_dim)   # (n, A, act_dim)
        return {
            f"drone_{i}": unflat[:, i, :]
            for i in range(NUM_DRONES)
        }

    def _flatten_rewards(self, reward_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        {"drone_i": (n,)} → (n, A) → (n*A,)
        All agents share the same reward so this is just repeat/reshape.
        """
        stacked = torch.stack(
            [reward_dict[f"drone_{i}"] for i in range(NUM_DRONES)],
            dim=1,
        )   # (n, A)
        return stacked.reshape(self.num_envs)   # (n*A,)

    def _flatten_dones(
        self,
        terminated_dict: dict[str, torch.Tensor],
        truncated_dict:  dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        terminated OR truncated → (n,) → repeat A times → (n*A,)
        All agents in an env share the same done signal (cooperative task).
        """
        # All agent dones are identical — just use drone_0
        done = (
            terminated_dict["drone_0"] | truncated_dict["drone_0"]
        ).float()   # (n,)

        # Repeat for each agent slot
        return done.unsqueeze(1).expand(-1, NUM_DRONES).reshape(self.num_envs)  # (n*A,)

    # ------------------------------------------------------------------
    # Passthrough attributes RSL-RL may query
    # ------------------------------------------------------------------

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    def get_observations(self):
        return self._flatten_obs(self.env._obs_manager.compute())

    def close(self):
        self.env.close()
