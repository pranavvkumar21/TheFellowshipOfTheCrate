from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# Goal sampling ranges — move to config later
# ---------------------------------------------------------------------------

GOAL_X_RANGE = (-1.0,  1.0)   # metres from env origin
GOAL_Y_RANGE = (-1.0,  1.0)
GOAL_Z_MIN   =  1.2            # minimum lift height
GOAL_Z_MAX   =  2.5            # maximum lift height

# How close the crate must be to count as success (metres)
GOAL_REACHED_THRESHOLD = 0.15


class CommandManager:
    """
    Samples and manages goal positions for each env.

    Goal is a 3D world position the crate should be lifted to.
    Resampled at every episode reset for the envs being reset.

    goal_pos_w shape: (num_envs, 3)
    """

    def __init__(self, env):
        self.env    = env
        self.device = env.device

        # Initialise goals for all envs
        self._goal_pos_w = torch.zeros(env.num_envs, 3, device=self.device)
        self._sample(torch.arange(env.num_envs, device=self.device))

        # Push initial goal into env so rewards/obs can use env._goal_pos_w
        self._sync_to_env()

        print(f"[CommandManager] goal X: {GOAL_X_RANGE}  "
              f"Y: {GOAL_Y_RANGE}  Z: [{GOAL_Z_MIN}, {GOAL_Z_MAX}]")
        print(f"[CommandManager] success threshold: {GOAL_REACHED_THRESHOLD} m")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        """Resample goals for the given envs and sync to env."""
        self._sample(env_ids)
        self._sync_to_env()

    @property
    def goal_pos_w(self) -> torch.Tensor:
        """Current goal positions. Shape: (num_envs, 3)"""
        return self._goal_pos_w

    def goal_reached(self) -> torch.Tensor:
        """
        Returns (num_envs,) bool — True if crate is within threshold of goal.
        """
        dist = torch.norm(
            self._goal_pos_w - self.env._crate.data.root_pos_w,
            dim=-1,
        )   # (n,)
        return dist < GOAL_REACHED_THRESHOLD

    def goal_rel_crate(self) -> torch.Tensor:
        """
        Goal position relative to crate. Shape: (num_envs, 3)
        Used directly as an observation term.
        """
        return self._goal_pos_w - self.env._crate.data.root_pos_w

    def goal_dist(self) -> torch.Tensor:
        """Euclidean distance from crate to goal. Shape: (num_envs,)"""
        return torch.norm(self.goal_rel_crate(), dim=-1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample(self, env_ids: torch.Tensor) -> None:
        """
        Sample new goal positions for env_ids.
        Goals are relative to each env's origin so they stay within the env.
        """
        n = len(env_ids)
        origins = self.env.scene.env_origins[env_ids]   # (n, 3)

        # Sample offsets uniformly within configured ranges
        offset = torch.zeros(n, 3, device=self.device)
        offset[:, 0].uniform_(GOAL_X_RANGE[0], GOAL_X_RANGE[1])
        offset[:, 1].uniform_(GOAL_Y_RANGE[0], GOAL_Y_RANGE[1])
        offset[:, 2].uniform_(GOAL_Z_MIN, GOAL_Z_MAX)

        self._goal_pos_w[env_ids] = origins + offset

    def _sync_to_env(self) -> None:
        """Write current goals into env._goal_pos_w so all managers see it."""
        self.env._goal_pos_w = self._goal_pos_w
