from __future__ import annotations
import torch
from quadcopter_lift_env_cfg import NUM_DRONES


class TerminationManager:

    def __init__(self, env):
        self.env    = env
        self.device = env.device

        self.drone_drone_radius: float = env.cfg.drone_collision_radius
        self.max_drone_height: float   = env.cfg.max_drone_height
        self.max_crate_tilt: float     = env.cfg.max_crate_tilt

        pairs = [(i, j) for i in range(NUM_DRONES) for j in range(i + 1, NUM_DRONES)]
        self._pair_idx = torch.tensor(pairs, device=self.device)   # (6, 2)

        self._episode_term_counts: dict[str, torch.Tensor] = {
            k: torch.zeros(env.num_envs, device=self.device)
            for k in [
                "inter_drone_collision",
                "drone_crate_contact",
                "drone_ground_contact",
                "crate_ground_contact",
                "crate_tip_over",
                "drone_too_high",
                "timeout",
            ]
        }
        self._prev_terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        print(f"[TerminationManager] drone_drone_radius = {self.drone_drone_radius} m")
        print(f"[TerminationManager] max_crate_tilt     = {self.max_crate_tilt:.3f} rad "
              f"({torch.tensor(self.max_crate_tilt).rad2deg().item():.1f}°)")
        print(f"[TerminationManager] drone pairs tracked: {len(pairs)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        logged_stats = {}
        for cause, count_tensor in self._episode_term_counts.items():
            logged_stats[f"term_{cause}"] = torch.mean(count_tensor[env_ids]).item()
            count_tensor[env_ids] = 0.0
        self._prev_terminated[env_ids] = False
        return logged_stats

    def compute(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        grace = self.env.episode_length_buf < self.env.cfg.reset_grace_steps

        inter_drone    = self._inter_drone_collision()
        drone_crate    = self._drone_crate_contact()
        drone_ground   = self._drone_ground_contact() #& ~grace
        crate_ground   = self._crate_ground_contact() #& ~grace
        drone_too_high = self._drone_too_high()
        crate_tip      = self._crate_tip_over()
        timeout        = self._timeout()

        terminated = inter_drone | drone_crate | drone_ground | drone_too_high | crate_tip
        timed_out  = timeout & ~terminated

        newly_terminated = terminated & ~self._prev_terminated
        self._log_new_terminations(
            newly_terminated, inter_drone, drone_crate,
            drone_ground, crate_ground, drone_too_high, crate_tip
        )
        self._prev_terminated = terminated.clone()

        return (
            {name: terminated for name in self.env.cfg.possible_agents},
            {name: timed_out  for name in self.env.cfg.possible_agents},
        )

    def get_termination_info(self) -> dict[str, torch.Tensor]:
        grace = self.env.episode_length_buf < self.env.cfg.reset_grace_steps
        return {
            "inter_drone_collision" : self._inter_drone_collision(),
            "drone_crate_contact"   : self._drone_crate_contact(),
            # "drone_ground_contact"  : self._drone_ground_contact() & ~grace,
            "crate_ground_contact"  : self._crate_ground_contact() & ~grace,
            "drone_too_high"        : self._drone_too_high(),
            "crate_tip_over"        : self._crate_tip_over(),
            "timeout"               : self._timeout(),
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_new_terminations(
        self,
        newly_terminated: torch.Tensor,
        inter_drone:    torch.Tensor,
        drone_crate:    torch.Tensor,
        drone_ground:   torch.Tensor,
        crate_ground:   torch.Tensor,
        drone_too_high: torch.Tensor,
        crate_tip:      torch.Tensor,
    ) -> None:
        term_ids = newly_terminated.nonzero(as_tuple=True)[0]
        for env_id in term_ids:
            if inter_drone[env_id]:
                self._episode_term_counts["inter_drone_collision"][env_id] += 1
            elif drone_crate[env_id]:
                self._episode_term_counts["drone_crate_contact"][env_id] += 1
            elif drone_ground[env_id]:
                self._episode_term_counts["drone_ground_contact"][env_id] += 1
            elif crate_ground[env_id]:
                self._episode_term_counts["crate_ground_contact"][env_id] += 1
            elif drone_too_high[env_id]:
                self._episode_term_counts["drone_too_high"][env_id] += 1
            elif crate_tip[env_id]:
                self._episode_term_counts["crate_tip_over"][env_id] += 1

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def _all_drone_pos(self) -> torch.Tensor:
        return torch.stack(
            [self.env._drones[f"drone_{i}"].data.root_pos_w for i in range(NUM_DRONES)],
            dim=1,
        )   # (n, 4, 3)

    def _inter_drone_collision(self) -> torch.Tensor:
        pos   = self._all_drone_pos()                          # (n, 4, 3)
        idx_a = self._pair_idx[:, 0]
        idx_b = self._pair_idx[:, 1]
        dist  = (pos[:, idx_a] - pos[:, idx_b]).norm(dim=-1)  # (n, 6)
        return (dist < self.drone_drone_radius).any(dim=-1)    # (n,)

    def _drone_crate_contact(self) -> torch.Tensor:
        any_contact = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        for i in range(NUM_DRONES):
            sensor = self.env._contact_sensors[f"drone_{i}_contact"]
            # (num_envs, num_bodies=1, num_filters, 3)
            force_mat = sensor.data.force_matrix_w              # uses filter_prim_paths_expr[web:19][web:25]
            if force_mat is None:
                continue
            crate_force = force_mat[:, 0, 0, :]                 # body 0, filter 0 = crate
            any_contact |= crate_force.norm(dim=-1) > 0.1
        return any_contact


    def _drone_ground_contact(self) -> torch.Tensor:
        any_contact = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        for i in range(NUM_DRONES):
            sensor = self.env._contact_sensors[f"drone_{i}_contact"]
            force_mat = sensor.data.force_matrix_w
            if force_mat is None or force_mat.shape[2] < 2:
                continue
            ground_force = force_mat[:, 0, 1, :]                # body 0, filter 1 = ground
            any_contact |= ground_force.norm(dim=-1) > 0.1
        return any_contact


    def _crate_ground_contact(self) -> torch.Tensor:
        # filter index 0 = ground (only filter on crate sensor)
        ground_force = self.env._crate_contact.data.net_forces_w[:, 0, :]   # (n, 3)
        return ground_force.norm(dim=-1) > 0.5   # higher threshold — crate is heavy

    def _drone_too_high(self) -> torch.Tensor:
        z = self._all_drone_pos()[:, :, 2]                         # (n, 4)
        return (z > self.max_drone_height).any(dim=-1)             # (n,)

    def _crate_tip_over(self) -> torch.Tensor:
        q          = self.env._crate.data.root_quat_w              # (n, 4) [w,x,y,z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        crate_up_z = 1.0 - 2.0 * (x * x + y * y)                 # cos(tilt)
        threshold  = torch.cos(torch.tensor(self.max_crate_tilt, device=self.device))
        return crate_up_z < threshold                              # (n,)

    def _timeout(self) -> torch.Tensor:
        return self.env.episode_length_buf >= self.env.max_episode_length - 1
