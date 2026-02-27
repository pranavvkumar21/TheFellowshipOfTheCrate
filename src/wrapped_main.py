# run_coop_lift_flattened.py
# Launches CoopLiftEnv wrapped in FlattenedMARLWrapper, runs N steps, 
# and prints observation/action/debug information.

from __future__ import annotations
import argparse
import time


# ---------------------------------------------------------------------------
# Isaac Lab app launch — MUST happen before any isaaclab/omni imports
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cooperative Crate Lift (Flattened Wrapper) — debug run")
parser.add_argument("--num_envs",   type=int, default=4,    help="Number of parallel environments")
parser.add_argument("--num_steps",  type=int, default=500,  help="Total simulation steps to run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True, livestream=2)
simulation_app = app_launcher.app

import torch
# ---------------------------------------------------------------------------
# Isaac Lab imports — AFTER app launch
# ---------------------------------------------------------------------------
from isaaclab.envs import DirectMARLEnv
from quadcopter_lift_env_cfg import CoopLiftEnvCfg, NUM_DRONES
from quadcopter_lift_env import CoopLiftEnv
from wrapper import FlattenedMARLWrapper


# ---------------------------------------------------------------------------
# Helper: generate sample actions
# ---------------------------------------------------------------------------
def sample_lift_actions_flat(num_envs_total: int, action_dim: int, device: str):
    """Small positive thrust delta for all (n*A) agents — drones lift slightly."""
    actions = torch.zeros(num_envs_total, action_dim, device=device)
    actions[:, 0] = 0.3  # positive thrust delta on first motor channel
    return actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = CoopLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 10.0

    print(f"\n{'='*60}")
    print("  Cooperative Crate Lift — Flattened Wrapper Debug Run")
    print(f"  num_envs     : {cfg.scene.num_envs}")
    print(f"  num_drones   : {NUM_DRONES}")
    print(f"  num_steps    : {args.num_steps}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ env setup
    base_env = CoopLiftEnv(cfg=cfg, render_mode=None)
    wrapped_env = FlattenedMARLWrapper(base_env)
    device = wrapped_env.device

    print(f"Wrapped env has {wrapped_env.num_envs} flattened agents "
          f"(({args.num_envs} envs × {NUM_DRONES} drones))")

    # ------------------------------------------------------------------ reset
    obs, info = wrapped_env.reset()
    print(f"\n--- Initial observation ---\n"
          f"  shape: {tuple(obs.shape)}\n"
          f"  sample[0, :5]: {obs[0, :5].cpu().numpy()}\n")

    # ------------------------------------------------------------------ loop
    step_times = []
    total_rewards = torch.zeros(wrapped_env.num_envs, device=device)

    for step in range(args.num_steps):
        t0 = time.perf_counter()

        # Sample lift actions for all flattened agents
        actions = sample_lift_actions_flat(
            wrapped_env.num_envs, wrapped_env.action_dim, device
        )

        # Step environment
        next_obs, rewards, dones, info = wrapped_env.step(actions)
        if step == 0:
            print(f"\n--- Step {step} ---")
            print(f"Actions shape: {actions.shape}")
            print(f"Rewards shape: {rewards.shape}")
            print(f"Dones shape: {dones.shape}")
            print(f"Sample actions[0, :5]: {actions[0, :5].cpu().numpy()}")
            print(f"Sample rewards[0]: {rewards[0].item():+.3f}")
            print(f"Sample dones[0]: {dones[0].item()}")
        dt = time.perf_counter() - t0
        step_times.append(dt)

        total_rewards += rewards

        # Debug info for first few steps
        if step < 10:
            print(f"[STEP {step:03d}] obs mean {obs.mean():+.3f} "
                  f"→ next mean {next_obs.mean():+.3f} | "
                  f"reward mean {rewards.mean():+.3f} | "
                  f"dones {dones.sum().item()} / {len(dones)}")

            # Crate and drone debug positions (only env 0)
            crate_z = base_env._crate.data.root_pos_w[0, 2].item()
            drone0_z = base_env._drones['drone_0'].data.root_pos_w[0, 2].item()
            rope_dist = drone0_z - 0.02 - (crate_z + base_env.cfg.crate_size[2]/2)
            print(f"         Crate Z: {crate_z:.3f}  Drone_0 Z: {drone0_z:.3f}  Rope dist: {rope_dist:.3f}")

        # Periodic stats output
        if step % 100 == 0 and step > 0:
            print(f"\n[Step {step}] avg reward {rewards.mean():+.4f} "
                  f"| min {rewards.min():+.4f} | max {rewards.max():+.4f}")
            print(f"           {dones.sum().item()} envs done this step")

        obs = next_obs

    # ------------------------------------------------------------------ summary
    avg_step_ms = sum(step_times) / len(step_times) * 1000
    print(f"\n{'='*60}")
    print(f"Run complete — {args.num_steps} steps")
    print(f"Avg step time : {avg_step_ms:.2f} ms  "
          f"({1000.0 / avg_step_ms:.1f} steps/s)")
    print(f"Mean cumulative reward per flattened agent: "
          f"{(total_rewards / args.num_steps).mean():+.4f}")
    print(f"{'='*60}\n")

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
