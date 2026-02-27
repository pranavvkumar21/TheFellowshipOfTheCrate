from __future__ import annotations
import torch
from quadcopter_lift_env_cfg import NUM_DRONES


class TerminationManager:
    """
    Termination conditions for cooperative lift.

    Tracks two separate signals per env:
        terminated: episode ends, counts as failure (crashed, collided, tipped)
        timed_out:  episode ends, does NOT count as failure (max steps reached)

    Conditions:
        1. Inter-drone collision     — any two drones closer than collision_radius
        2. Drone-crate collision     — any drone closer than drone_crate_radius to crate surface
        3. Drone-ground collision    — any drone below min_drone_height
        4. Crate tip-over            — crate orientation deviates beyond max_crate_tilt
        5. Timeout                   — episode_length_buf >= max_episode_length - 1
    """

    def __init__(self, env):
        self.env    = env
        self.device = env.device

        # Collision radii
        self.drone_drone_radius: float = env.cfg.drone_collision_radius   # e.g. 0.15 m
        self.drone_crate_radius: float = env.cfg.drone_crate_radius       # e.g. 0.20 m

        # Height bounds
        self.min_drone_height: float = env.cfg.min_drone_height           # e.g. 0.05 m
        self.max_drone_height: float = env.cfg.max_drone_height           # e.g. 5.0  m

        # Crate tilt limit (radians) — angle between crate up-axis and world up-axis
        self.max_crate_tilt: float = env.cfg.max_crate_tilt               # e.g. 0.52 rad (30°)

        # Precompute all unique drone pairs — shape (num_pairs, 2)
        # For 4 drones: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) → 6 pairs
        pairs = [(i, j) for i in range(NUM_DRONES) for j in range(i + 1, NUM_DRONES)]
        self._pair_idx = torch.tensor(pairs, device=self.device)   # (num_pairs, 2)

        print(f"[TerminationManager] drone_drone_radius = {self.drone_drone_radius} m")
        print(f"[TerminationManager] drone_crate_radius = {self.drone_crate_radius} m")
        print(f"[TerminationManager] max_crate_tilt     = {self.max_crate_tilt:.3f} rad "
              f"({torch.tensor(self.max_crate_tilt).rad2deg().item():.1f}°)")
        print(f"[TerminationManager] drone pairs tracked: {len(pairs)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Returns:
            terminated: {"drone_i": (num_envs,) bool} — failure termination
            timed_out:  {"drone_i": (num_envs,) bool} — timeout termination
        """
        terminated = (
              self._inter_drone_collision()   # (n,) bool
            # | self._drone_crate_aabb()       # (n,) bool
            | self._drone_ground_collision()
            | self._crate_tip_over()
        )

        timed_out = self._timeout()           # (n,) bool

        return (
            {name: terminated for name in self.env.cfg.possible_agents},
            {name: timed_out  for name in self.env.cfg.possible_agents},
        )

    def get_termination_info(self) -> dict[str, torch.Tensor]:
        """
        Returns individual condition tensors for logging/reward shaping.
        Each is shape (num_envs,) bool.
        """
        return {
            "inter_drone_collision" : self._inter_drone_collision(),
            "drone_crate_collision" : self._drone_crate_collision(),
            "drone_ground_collision": self._drone_ground_collision(),
            "crate_tip_over"        : self._crate_tip_over(),
            "timeout"               : self._timeout(),
        }

    # ------------------------------------------------------------------
    # Termination conditions — each returns (num_envs,) bool
    # ------------------------------------------------------------------

    def _all_drone_pos(self) -> torch.Tensor:
        """Stack all drone positions. Shape: (n, A, 3)"""
        return torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_pos_w
             for i in range(NUM_DRONES)],
            dim=1,
        )   # (n, A, 3)

    def _inter_drone_collision(self) -> torch.Tensor:
        """
        True if any pair of drones are within drone_drone_radius of each other.

        Pairwise distances: (n, num_pairs)
        Uses precomputed pair index to avoid full (A x A) matrix.
        """
        pos = self._all_drone_pos()   # (n, A, 3)

        # Gather positions for each side of every pair
        idx_a = self._pair_idx[:, 0]   # (num_pairs,)
        idx_b = self._pair_idx[:, 1]

        pos_a = pos[:, idx_a, :]       # (n, num_pairs, 3)
        pos_b = pos[:, idx_b, :]       # (n, num_pairs, 3)

        dist = (pos_a - pos_b).norm(dim=-1)   # (n, num_pairs)

        # Any pair below threshold → collision in that env
        return (dist < self.drone_drone_radius).any(dim=-1)   # (n,)

    def _drone_crate_collision(self) -> torch.Tensor:
        """
        True if any drone centre is within drone_crate_radius of the
        crate centre. Uses a simple sphere approximation around crate CoM
        — good enough for termination, not for exact surface contact.

        For a tighter AABB check, use _drone_crate_aabb() below instead.
        """
        pos       = self._all_drone_pos()                              # (n, A, 3)
        crate_pos = self.env._crate.data.root_pos_w.unsqueeze(1)      # (n, 1, 3)

        dist = (pos - crate_pos).norm(dim=-1)   # (n, A)

        return (dist < self.drone_crate_radius).any(dim=-1)            # (n,)

    def _drone_crate_aabb(self) -> torch.Tensor:
        """
        Tighter alternative to sphere check: axis-aligned bounding box
        using actual randomised crate_size + a small margin.
        True if any drone centre is inside the inflated crate AABB.
        """
        pos       = self._all_drone_pos()                         # (n, A, 3)
        crate_pos = self.env._crate.data.root_pos_w               # (n, 3)
        half      = self.env._crate_size / 2                      # (n, 3)
        margin    = 0.05                                           # 5 cm safety margin

        # Expand crate centre and half-extents to match drone shape
        crate_pos_e = crate_pos.unsqueeze(1)                      # (n, 1, 3)
        half_e      = (half + margin).unsqueeze(1)                # (n, 1, 3)

        # Check if drone is inside box on all 3 axes simultaneously
        inside = ((pos - crate_pos_e).abs() < half_e).all(dim=-1)  # (n, A)

        return inside.any(dim=-1)                                  # (n,)

    def _drone_ground_collision(self) -> torch.Tensor:
        """
        True if any drone goes below min_drone_height or above max_drone_height.
        """
        pos = self._all_drone_pos()    # (n, A, 3)
        z   = pos[:, :, 2]            # (n, A)

        too_low  = (z < self.min_drone_height).any(dim=-1)   # (n,)
        too_high = (z > self.max_drone_height).any(dim=-1)   # (n,)

        return too_low | too_high                             # (n,)

    def _crate_tip_over(self) -> torch.Tensor:
        """
        True if the crate has tipped beyond max_crate_tilt.

        Method: extract the crate's local up-axis (rotated world Z)
        from its quaternion, then measure the angle against world Z.

        quat convention: (w, x, y, z)
        Rotate world-up [0,0,1] by crate quaternion to get crate-up,
        then cos(tilt) = dot(crate_up, world_up) = crate_up[:, 2]
        """
        q = self.env._crate.data.root_quat_w    # (n, 4)  [w, x, y, z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Rotate world Z-axis [0,0,1] by quaternion:
        # crate_up_z = 1 - 2*(x² + y²)   (z-component of rotated Z)
        crate_up_z = 1.0 - 2.0 * (x * x + y * y)   # (n,)  = cos(tilt_angle)

        # cos(tilt) < cos(max_tilt) → tilt > max_tilt
        tilt_threshold = torch.cos(
            torch.tensor(self.max_crate_tilt, device=self.device)
        )

        return crate_up_z < tilt_threshold   # (n,)

    def _timeout(self) -> torch.Tensor:
        """True if episode has reached max steps."""
        return self.env.episode_length_buf >= self.env.max_episode_length - 1  # (n,)
