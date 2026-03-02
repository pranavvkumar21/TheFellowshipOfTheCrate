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
    _TERM_REGISTRY = [
        ("drone_quat",             4),
        ("drone_linvel",           3),
        ("crate_rel_pos",          3),
        ("crate_height",           1),   # absolute Z of crate
        ("crate_abs_vel",          3),
        ("crate_quat",             4),
        ("goal_rel_pos",           3),
        ("goal_abs_vel_error",     3),
        ("action_state",           4),
        ("neighbour_rel_pos",      (NUM_DRONES-1)*3),
        ("neighbour_linvel",       (NUM_DRONES-1)*3),
        ("neighbour_quat",         (NUM_DRONES-1)*4),
        ("neighbour_action_state", (NUM_DRONES-1)*4),
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
        for name, tensor in [
            ("drone_quat",             self._drone_quat()),
            ("drone_linvel",           self._drone_linvel()),
            ("crate_rel_pos",          self._crate_rel_pos()),
            ("crate_height",           self._crate_height()),
            ("crate_abs_vel",          self._crate_abs_vel()),
            ("crate_quat",             self._crate_quat()),
            ("goal_rel_pos",           self._goal_rel_pos()),
            ("goal_abs_vel_error",     self._goal_abs_vel_error()),
            ("action_state",           self._action_state()),
            ("neighbour_rel_pos",      self._neighbour_rel_pos()),
            ("neighbour_linvel",       self._neighbour_linvel()),
            ("neighbour_quat",         self._neighbour_quat()),
            ("neighbour_action_state", self._neighbour_action_state()),
        ]:
            if not tensor.isfinite().all():
                print(f"BAD TERM: {name} — min={tensor.min():.3f} max={tensor.max():.3f}")

        full_obs = torch.cat([
            self._drone_quat(),              # 4
            self._drone_linvel(),            # 3
            self._crate_rel_pos(),           # 3
            self._crate_height(),            # 1  ← right after crate_rel_pos
            self._crate_abs_vel(),           # 3
            self._crate_quat(),              # 4
            self._goal_rel_pos(),            # 3
            self._goal_abs_vel_error(),      # 3
            self._action_state(),            # 4  ← before neighbours
            self._neighbour_rel_pos(),       # 9
            self._neighbour_linvel(),        # 9
            self._neighbour_quat(),          # 12
            self._neighbour_action_state(),  # 12
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
        all_pos = self._all_drone_pos()  # (n, A, 3)
        A = NUM_DRONES
        pairwise = all_pos.unsqueeze(2) - all_pos.unsqueeze(1)  # (n, A, A, 3)
        result = []
        for i in range(A):
            others = [j for j in range(A) if j != i]
            result.append(pairwise[:, i, others, :].reshape(pairwise.shape[0], (A-1)*3))
        return torch.stack(result, dim=1)  # (n, A, (A-1)*3)

    def _action_state(self) -> torch.Tensor:
        # (n, A, 4) — normalised so thrust in [0,1], torques in [-1,1]
        return self.env._action_manager.get_state_normalised()
    def _neighbour_linvel(self) -> torch.Tensor:
        all_vel = self._drone_linvel()  # (n, A, 3)
        A = NUM_DRONES
        result = []
        for i in range(A):
            others = [j for j in range(A) if j != i]
            result.append(all_vel[:, others, :].reshape(all_vel.shape[0], (A-1)*3))
        return torch.stack(result, dim=1)  # (n, A, (A-1)*3)

    def _neighbour_quat(self) -> torch.Tensor:
        all_quat = self._drone_quat()  # (n, A, 4)
        A = NUM_DRONES
        result = []
        for i in range(A):
            others = [j for j in range(A) if j != i]
            result.append(all_quat[:, others, :].reshape(all_quat.shape[0], (A-1)*4))
        return torch.stack(result, dim=1)  # (n, A, (A-1)*4)
    def _neighbour_action_state(self) -> torch.Tensor:
        action_state = self.env._action_manager.get_state_normalised()  # (n, A, 4)
        A = NUM_DRONES
        
        # Same pattern as neighbour_linvel
        action_broadcast = action_state.unsqueeze(1).expand(-1, A, -1, -1)  # (n, A, A, 4)
        
        result = []
        for i in range(A):
            others = [j for j in range(A) if j != i]
            result.append(action_broadcast[:, i, others, :].reshape(action_state.shape[0], (A-1)*4))
        return torch.stack(result, dim=1)  # (n, A, (A-1)*4)
    def _crate_height(self) -> torch.Tensor:
        z = self.env._crate.data.root_pos_w[:, 2:3]  # (n, 1)
        return self._broadcast_crate(z)               # (n, A, 1)