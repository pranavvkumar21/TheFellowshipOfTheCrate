# run_coop_lift.py
# Launches CoopLiftEnv with 4 envs × 4 drones, runs N steps with random actions,
# and prints per-step reward stats.

from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# Isaac Lab app launch — MUST happen before any isaaclab/omni imports
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cooperative Crate Lift — random policy smoke test")
parser.add_argument("--num_envs",   type=int, default=4,    help="Number of parallel environments")
parser.add_argument("--num_steps",  type=int, default=2000, help="Total simulation steps to run")
# parser.add_argument("--headless",   action="store_true",    help="Run without GUI")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True, livestream=2)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All other imports AFTER the app is launched
# ---------------------------------------------------------------------------

import time
import torch
from isaaclab.envs import DirectMARLEnv

from quadcopter_lift_env_cfg import CoopLiftEnvCfg, NUM_DRONES
from quadcopter_lift_env     import CoopLiftEnv


# ---------------------------------------------------------------------------
# Helper: sample random actions for all agents
# ---------------------------------------------------------------------------
def sample_random_actions(
    env: CoopLiftEnv,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Returns random actions in [-1, 1] where all agents in the same env
    receive identical actions — same thrust and moment for every drone.
    Shape per agent: (num_envs, action_dim)
    """
    # Sample once per env, then broadcast to all agents
    shared = torch.rand(env.num_envs, env.cfg.action_spaces["drone_0"], device=device) * 2.0 - 1.0
    return {name: shared.clone() for name in env.cfg.possible_agents}

def sample_hover_actions(env, device):
    """
    Zero delta actions — residual system holds at hover thrust.
    Drones should hover in place, crate should hang still.
    """
    zeros = torch.zeros(env.num_envs, env.cfg.action_spaces["drone_0"], device=device)
    return {name: zeros for name in env.cfg.possible_agents}


def sample_lift_actions(env, device):
    """
    Small positive thrust delta — all drones push up slightly.
    Crate should slowly rise.
    """
    act = torch.zeros(env.num_envs, env.cfg.action_spaces["drone_0"], device=device)
    act[:, 0] = 0.3   # positive thrust delta only
    return {name: act for name in env.cfg.possible_agents}


# ---------------------------------------------------------------------------
# Helper: pretty-print reward stats
# ---------------------------------------------------------------------------
def print_reward_stats(
    step: int,
    rewards: dict[str, torch.Tensor],
    interval: int = 100,
) -> None:
    if step % interval != 0:
        return

    lines = [f"\n[Step {step:>5d}] Rewards"]
    for name, rew in rewards.items():
        lines.append(
            f"  {name}: mean={rew.mean().item():+.4f}  "
            f"min={rew.min().item():+.4f}  "
            f"max={rew.max().item():+.4f}"
        )
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Helper: print episode termination counts
# ---------------------------------------------------------------------------
def print_done_stats(
    step: int,
    terminated: dict[str, torch.Tensor],
    time_outs:  dict[str, torch.Tensor],
    interval: int = 100,
) -> None:
    if step % interval != 0:
        return

    # All agents share the same terminated tensor — just check drone_0
    n_term    = terminated["drone_0"].sum().item()
    n_timeout = time_outs["drone_0"].sum().item()
    print(f"           Terminated: {int(n_term):>3d}  |  Timed-out: {int(n_timeout):>3d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------ cfg
    cfg = CoopLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs     # override from CLI (default 4)
    cfg.episode_length_s = 10.0

    print(f"\n{'='*60}")
    print(f"  Cooperative Crate Lift — smoke test")
    print(f"  num_envs   : {cfg.scene.num_envs}")
    print(f"  num_agents : {NUM_DRONES}  {cfg.possible_agents}")
    print(f"  steps      : {args.num_steps}")
    print(f"  action_dim : {cfg.action_spaces['drone_0']}")
    print(f"  obs_dim    : {cfg.observation_spaces['drone_0']}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ env
    env = CoopLiftEnv(cfg=cfg, render_mode=None if args.headless else "human")
    device = env.device

    # ------------------------------------------------------------------ reset
    obs_dict, info = env.reset()

    # Sanity-check observation shapes
    print("--- Observation shapes after reset ---")
    for name, obs in obs_dict.items():
        print(f"  {name}: {tuple(obs.shape)}")
    print()

    # ------------------------------------------------------------------ loop
    total_rewards = {name: 0.0 for name in cfg.possible_agents}
    step_times: list[float] = []

    for step in range(args.num_steps):
        t0 = time.perf_counter()

        # Random policy
        actions = sample_lift_actions(env, device)

        # Step the environment
        obs_dict, reward_dict, terminated_dict, time_outs_dict, info = env.step(actions)

        step_times.append(time.perf_counter() - t0)

        # Accumulate rewards
        for name in cfg.possible_agents:
            crate_pos = env._crate.data.root_pos_w  # Debug: print crate and drone positions for first 10 steps
        if step < 10:
            crate_pos = env._crate.data.root_pos_w[:, :3]
            drone_pos = env._drones['drone_0'].data.root_pos_w[:, :3]
            # Only print env 0
            rope_attach_dist = drone_pos[0, 2] - 0.02 - (crate_pos[0, 2] + env.cfg.crate_size[2]/2)
            print(f"[DEBUG STEP {step}] Crate Z: {crate_pos[0, 2]:.3f}  Drone_0 Z: {drone_pos[0, 2]:.3f}  Rope dist: {rope_attach_dist:.3f}")

        # Periodic console output
        print_reward_stats(step, reward_dict, interval=100)
        print_done_stats(step, terminated_dict, time_outs_dict, interval=100)

    # ------------------------------------------------------------------ summary
    avg_step_ms = (sum(step_times) / len(step_times)) * 1000.0

    print(f"\n{'='*60}")
    print(f"  Run complete — {args.num_steps} steps")
    print(f"  Avg step time : {avg_step_ms:.2f} ms  "
          f"({1000.0 / avg_step_ms:.0f} steps/s)")
    print(f"\n  Cumulative mean reward per agent:")
    for name, total in total_rewards.items():
        print(f"    {name}: {total / args.num_steps:+.4f} avg/step")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
