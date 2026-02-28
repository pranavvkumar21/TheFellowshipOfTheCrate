from __future__ import annotations
import torch
from quadcopter_lift_env_cfg import NUM_DRONES


_CORNER_SIGNS = torch.tensor([
    [ 1.,  1.],
    [-1.,  1.],
    [-1., -1.],
    [ 1., -1.],
], dtype=torch.float32)  # (A, 2)


class ObservationManager:
    """
    Computes per-drone observations, fully vectorised.

    Each drone observes independently — no agent sees another drone's state.
    All positions are expressed relative to that drone's own position,
    except where noted as absolute.

    Output shape per forward(): (num_envs, NUM_DRONES, obs_dim)
    Individual agent slice:     (num_envs, obs_dim)  → full_obs[:, i, :]
    """

    # Registry: (name, feature_dim)
    # Order here defines concatenation order in the final obs vector.
    _TERM_REGISTRY: list[tuple[str, int]] = [
        ("drone_pos",           3),   # drone world position (absolute)
        ("drone_quat",          4),   # drone orientation (world frame)
        ("drone_linvel",        3),   # drone linear velocity (world frame)
        ("crate_rel_pos",       3),   # crate pos relative to this drone
        ("crate_abs_vel",       3),   # crate linear velocity (world frame, absolute)
        ("crate_quat",          4),   # crate orientation (world frame)
        ("goal_rel_pos",        3),   # goal pos relative to crate (command error)
        ("goal_abs_vel_error",  3),   # crate velocity in direction of goal (abs)
        ("neighbour_rel_pos", (NUM_DRONES - 1) * 3),  # relative pos of other drones
        ("neighbour_linvel",   (NUM_DRONES - 1) * 3),  # absolute linvel of other drones
        ("neighbour_quat",     (NUM_DRONES - 1) * 4),  # absolute orientation of other drones
    ]

    def __init__(self, env):
        self.env    = env
        self.device = env.device

        # Precompute obs_dim once at setup time
        self.obs_dim: int = sum(dim for _, dim in self._TERM_REGISTRY)

        # Cache corner signs on device
        self._corner_signs = _CORNER_SIGNS.to(self.device)  # (A, 2)

        print(f"[ObservationManager] obs_dim = {self.obs_dim}")
        print(f"[ObservationManager] terms:")
        offset = 0
        for name, dim in self._TERM_REGISTRY:
            print(f"  [{offset:2d}:{offset+dim:2d}]  {name}  ({dim})")
            offset += dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of obs for a single agent in a single env: (obs_dim,)"""
        return (self.obs_dim,)

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Returns per-agent obs dict: {"drone_i": (num_envs, obs_dim)}
        Internally computes (num_envs, NUM_DRONES, obs_dim) then splits.
        """
        full_obs = torch.cat([
            self._drone_pos(),           # (n, A, 3)
            self._drone_quat(),          # (n, A, 4)
            self._drone_linvel(),        # (n, A, 3)
            self._crate_rel_pos(),       # (n, A, 3)
            self._crate_abs_vel(),       # (n, A, 3)
            self._crate_quat(),          # (n, A, 4)
            self._goal_rel_pos(),        # (n, A, 3)
            self._goal_abs_vel_error(),  # (n, A, 3)
            self._neighbour_rel_pos(),   # (n, A, (A-1)*3)
            self._neighbour_linvel(),     # (n, A, (A-1)*3)
            self._neighbour_quat(),       # (n, A, (A-1)*4)
        ], dim=-1)  # (n, A, obs_dim)

        return {
            f"drone_{i}": full_obs[:, i, :]
            for i in range(NUM_DRONES)
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_drone_pos(self) -> torch.Tensor:
        """Stacks all drone positions. Shape: (n, A, 3)"""
        return torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_pos_w
             for i in range(NUM_DRONES)],
            dim=1,
        )

    def _broadcast_crate(self, x: torch.Tensor) -> torch.Tensor:
        """(n, K) → (n, A, K) broadcast to all agents."""
        return x.unsqueeze(1).expand(-1, NUM_DRONES, -1)

    # ------------------------------------------------------------------
    # Observation terms — each returns (n, A, K)
    # ------------------------------------------------------------------

    def _drone_pos(self) -> torch.Tensor:
        """Absolute drone world position. (n, A, 3)"""
        return self._all_drone_pos()

    def _drone_quat(self) -> torch.Tensor:
        """Drone orientation quaternion (w, x, y, z) in world frame. (n, A, 4)"""
        return torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_quat_w
             for i in range(NUM_DRONES)],
            dim=1,
        )

    def _drone_linvel(self) -> torch.Tensor:
        """Drone linear velocity in world frame. (n, A, 3)"""
        return torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_lin_vel_w
             for i in range(NUM_DRONES)],
            dim=1,
        )

    def _crate_rel_pos(self) -> torch.Tensor:
        """
        Crate position relative to each drone's own position.
        = crate_pos_world - drone_pos_world
        (n, A, 3)
        """
        crate_pos  = self._broadcast_crate(self.env._crate.data.root_pos_w)  # (n, A, 3)
        drone_pos  = self._all_drone_pos()                                    # (n, A, 3)
        return crate_pos - drone_pos

    def _crate_abs_vel(self) -> torch.Tensor:
        """
        Crate absolute linear velocity in world frame, broadcast to all agents.
        (n, A, 3)
        """
        return self._broadcast_crate(self.env._crate.data.root_lin_vel_w)

    def _crate_quat(self) -> torch.Tensor:
        """
        Crate orientation quaternion (w, x, y, z) in world frame,
        broadcast to all agents. (n, A, 4)
        """
        return self._broadcast_crate(self.env._crate.data.root_quat_w)

    def _goal_rel_pos(self) -> torch.Tensor:
        """
        Command error: goal position relative to current crate position.
        = goal_pos_world - crate_pos_world
        Tells each drone how far the crate still needs to travel.
        Broadcast to all agents. (n, A, 3)
        """
        goal_pos  = self.env._goal_pos_w                          # (n, 3)
        crate_pos = self.env._crate.data.root_pos_w               # (n, 3)
        rel       = goal_pos - crate_pos                          # (n, 3)
        return self._broadcast_crate(rel)                         # (n, A, 3)

    def _goal_abs_vel_error(self) -> torch.Tensor:
        """
        Crate velocity expressed as progress toward the goal:
        desired_vel = normalised(goal - crate_pos) * ||crate_linvel||
        vel_error   = desired_vel - actual_crate_linvel
        This tells each drone whether the crate is moving toward the goal
        or drifting sideways. Broadcast to all agents. (n, A, 3)
        """
        goal_pos   = self.env._goal_pos_w                         # (n, 3)
        crate_pos  = self.env._crate.data.root_pos_w              # (n, 3)
        crate_vel  = self.env._crate.data.root_lin_vel_w          # (n, 3)

        to_goal     = goal_pos - crate_pos                        # (n, 3)
        dist        = to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir    = to_goal / dist                              # (n, 3) unit vec

        speed       = crate_vel.norm(dim=-1, keepdim=True)        # (n, 1)
        desired_vel = goal_dir * speed                            # (n, 3)
        vel_error   = desired_vel - crate_vel                     # (n, 3)

        return self._broadcast_crate(vel_error)                   # (n, A, 3)
    def _neighbour_rel_pos(self) -> torch.Tensor:
        all_pos = self._all_drone_pos()   # (n, A, 3)

        # Build (n, A, A, 3): all pairwise differences
        pairwise = all_pos.unsqueeze(2) - all_pos.unsqueeze(1)   # (n, A, A, 3)

        # Mask out self (diagonal) and flatten remaining A-1 neighbours
        A = NUM_DRONES
        mask = ~torch.eye(A, dtype=torch.bool, device=self.device)   # (A, A)
        
        # FIX: Apply the 2D mask directly to dims 1 and 2
        return pairwise[:, mask].reshape(pairwise.shape[0], A, (A - 1) * 3)
    def _neighbour_linvel(self) -> torch.Tensor:
        """
        Absolute linear velocity of all other drones.
        Broadcasts (n, A, 3) to (n, A, A, 3) so that for each observer drone 'i' (dim 1), 
        we have the velocities of all target drones 'j' (dim 2).
        Returns: (n, A, (A-1)*3)
        """
        all_vel = self._drone_linvel()    # (n, A, 3)
        A = NUM_DRONES
        
        # Expand observer to dimension 1: (n, 1, A, 3) -> (n, A, A, 3)
        vel_broadcast = all_vel.unsqueeze(1).expand(-1, A, -1, -1)
        
        # Mask out self
        mask = ~torch.eye(A, dtype=torch.bool, device=self.device)  # (A, A)
        
        # Apply mask and reshape
        return vel_broadcast[:, mask].reshape(all_vel.shape[0], A, (A - 1) * 3)

    def _neighbour_quat(self) -> torch.Tensor:
        """
        Orientation (quaternion) of all other drones.
        Returns: (n, A, (A-1)*4)
        """
        all_quat = self._drone_quat()     # (n, A, 4)
        A = NUM_DRONES
        
        # Expand observer to dimension 1: (n, 1, A, 4) -> (n, A, A, 4)
        quat_broadcast = all_quat.unsqueeze(1).expand(-1, A, -1, -1)
        
        # Mask out self
        mask = ~torch.eye(A, dtype=torch.bool, device=self.device)  # (A, A)
        
        return quat_broadcast[:, mask].reshape(all_quat.shape[0], A, (A - 1) * 4)
 