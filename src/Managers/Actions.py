from __future__ import annotations
import torch
from quadcopter_lift_env_cfg import NUM_DRONES


class ActionManager:
    """
    Residual action system for cooperative lift.

    Each drone outputs a 4-dim action: [delta_thrust, delta_mx, delta_my, delta_mz]
    These are INCREMENTS added to the current thrust/torque state.

    Residual formulation:
        thrust[t+1] = clip(thrust[t] + delta_thrust * thrust_scale, 0, max_thrust)
        torque[t+1] = clip(torque[t] + delta_torque * torque_scale, -max_torque, max_torque)

    Benefits over direct action:
        - Smoother control: policy learns corrections, not absolute setpoints
        - Easier exploration: small deltas around hover are physically meaningful
        - Implicit temporal smoothing: large jumps are impossible in one step

    Action space per drone: 4  → [delta_thrust, delta_mx, delta_my, delta_mz]
    Internal state shape:   (num_envs, NUM_DRONES, 4)
                             ↑ thrust is dim 0, torques are dims 1-3
    """

    # Registry: (name, slice, description)
    _TERM_REGISTRY = [
        ("delta_thrust", slice(0, 1), "vertical thrust increment"),
        ("delta_mx",     slice(1, 2), "roll torque increment"),
        ("delta_my",     slice(2, 3), "pitch torque increment"),
        ("delta_mz",     slice(3, 4), "yaw torque increment"),
    ]
    ACTION_DIM = 4

    def __init__(self, env):
        self.env    = env
        self.device = env.device
        self.num_envs = env.num_envs

        # Scaling factors
        self._hover_thrust = env._drone_weight          # N — weight of one drone
        self._max_thrust   = env.cfg.thrust_to_weight * env._drone_weight
        self._torque_scale = env.cfg.moment_scale
        self._max_torque   = env.cfg.moment_scale * 1.0  # clip at ±1 scaled unit

        # Delta scales: how much one unit of network output changes the state
        # Thrust delta: fraction of hover thrust per step
        self._thrust_delta_scale = env.cfg.thrust_delta_scale   # e.g. 0.05
        # Torque delta: fraction of max torque per step
        self._torque_delta_scale = env.cfg.torque_delta_scale   # e.g. 0.05

        # Internal residual state: (num_envs, NUM_DRONES, 4)
        # [:, :, 0] = current thrust per drone per env
        # [:, :, 1:4] = current torques per drone per env
        self._state = torch.zeros(self.num_envs, NUM_DRONES, 4, device=self.device)

        # Initialise thrust at hover so drones don't immediately fall
        self._state[:, :, 0] = self._hover_thrust

        print(f"[ActionManager] action_dim = {self.ACTION_DIM} per drone")
        print(f"[ActionManager] hover_thrust     = {self._hover_thrust:.3f} N")
        print(f"[ActionManager] max_thrust       = {self._max_thrust:.3f} N")
        print(f"[ActionManager] thrust_delta_scale = {self._thrust_delta_scale}")
        print(f"[ActionManager] torque_delta_scale = {self._torque_delta_scale}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset residual state to hover thrust for given envs."""
        self._state[env_ids] = 0.0
        self._state[env_ids, :, 0] = self._hover_thrust

    def step(self, actions: dict[str, torch.Tensor]) -> None:
        """
        Apply action increments and update internal residual state.

        Args:
            actions: {"drone_i": (num_envs, 4)} — raw network outputs in [-1, 1]
        """
        # Stack per-agent actions → (num_envs, NUM_DRONES, 4)
        raw = torch.stack(
            [actions[f"drone_{i}"] for i in range(NUM_DRONES)],
            dim=1,
        ).clamp(-1.0, 1.0)   # (n, A, 4)

        # Compute deltas
        delta_thrust = raw[:, :, 0:1] * self._thrust_delta_scale   # (n, A, 1)
        delta_torque = raw[:, :, 1:4] * self._torque_delta_scale   # (n, A, 3)

        # Update residual state with clipping
        self._state[:, :, 0:1] = (
            self._state[:, :, 0:1] + delta_thrust
        ).clamp(0.0, self._max_thrust)

        self._state[:, :, 1:4] = (
            self._state[:, :, 1:4] + delta_torque
        ).clamp(-self._max_torque, self._max_torque)

    def get_forces_and_torques(self) -> tuple[dict[str, torch.Tensor],
                                               dict[str, torch.Tensor]]:
        """
        Returns per-drone thrust and torque dicts ready for
        set_external_force_and_torque().

        Forces shape:  {"drone_i": (num_envs, 1, 3)}  — z-axis thrust only
        Torques shape: {"drone_i": (num_envs, 1, 3)}
        """
        thrust_z = self._state[:, :, 0]    # (n, A)
        torques   = self._state[:, :, 1:4] # (n, A, 3)

        forces_dict  = {}
        torques_dict = {}
        for i in range(NUM_DRONES):
            name = f"drone_{i}"
            # Force: only z component, shape (n, 1, 3)
            f = torch.zeros(self.num_envs, 1, 3, device=self.device)
            f[:, 0, 2] = thrust_z[:, i]
            forces_dict[name] = f

            # Torque: shape (n, 1, 3)
            torques_dict[name] = torques[:, i:i+1, :]   # (n, 1, 3)

        return forces_dict, torques_dict

    def get_state(self) -> torch.Tensor:
        """
        Current residual action state. Useful as obs input to policy.
        Shape: (num_envs, NUM_DRONES, 4)
        """
        return self._state.clone()

    def get_state_normalised(self) -> torch.Tensor:
        """
        State normalised to [-1, 1] — use this if feeding current action
        state back into the observation vector.
        Shape: (num_envs, NUM_DRONES, 4)
        """
        norm = self._state.clone()
        norm[:, :, 0:1] = norm[:, :, 0:1] / self._max_thrust          # thrust → [0, 1]
        norm[:, :, 1:4] = norm[:, :, 1:4] / (self._max_torque + 1e-6) # torque → [-1, 1]
        return norm
