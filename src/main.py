#!/usr/bin/env python3
"""
train.py — RSL-RL training & evaluation entry-point for CoopLift.

Usage
-----
# Train (new run)
python train.py --mode train --num_envs 256

# Train (resume latest checkpoint)
python train.py --mode train --load --num_envs 256

# Eval (latest checkpoint, record video)
python train.py --mode eval --checkpoint logs/coop_lift/run_01/model_5000.pt

Notes
-----
- The FlattenedMARLWrapper exposes (num_envs * NUM_DRONES, obs_dim) to RSL-RL.
- All per-agent rewards are identical (shared team reward), so the value
  function learns a single team critic.
- Reward component sums and termination counts are logged to TensorBoard
  every `log_interval` iterations via a custom callback on the runner's
  writer.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI — parsed before Isaac Sim app launch
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser(description="CoopLift RSL-RL trainer / evaluator")
p.add_argument("--mode",       choices=["train", "eval"], default="train")
p.add_argument("--num_envs",   type=int, default=256,
               help="Number of parallel Isaac Lab environments")
p.add_argument("--load",       action="store_true",
               help="Resume from the latest checkpoint in the experiment folder")
p.add_argument("--checkpoint", type=str, default=None,
               help="Explicit path to a .pt file to load (overrides --load)")
p.add_argument("--episode_s",  type=float, default=15.0,
               help="Episode length in seconds")
p.add_argument("--log_interval", type=int, default=10,
               help="Log reward/termination stats every N iterations")
p.add_argument("--eval_steps",  type=int, default=500,
               help="Number of sim steps to record during eval")
args = p.parse_args()

# ---------------------------------------------------------------------------
# Isaac Sim app — MUST be launched before any omni / isaaclab imports
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

if args.mode == "train":
    livestream      = 0
    enable_cameras  = False
else:
    livestream      = 2
    enable_cameras  = True

simulation_app = AppLauncher(
    headless=True,
    livestream=livestream,
    enable_cameras=enable_cameras,
).app

# ---------------------------------------------------------------------------
# Remaining imports (after sim launch)
# ---------------------------------------------------------------------------
import torch
import numpy as np

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from quadcopter_lift_env_cfg import CoopLiftEnvCfg, NUM_DRONES
from quadcopter_lift_env     import CoopLiftEnv
from wrapper                 import FlattenedMARLWrapper
from runner_cfg              import create_runner_cfg
from registration            import register_envs
from utils.env_info          import print_env_info
ROOT = Path(__file__).resolve().parent


# ===========================================================================
# Helpers
# ===========================================================================

def _latest_checkpoint(experiment_dir: Path) -> str | None:
    """Return path to the most recently created model_*.pt file, or None."""
    pts = list(experiment_dir.glob("model_*.pt"))
    if not pts:
        return None
    return str(max(pts, key=os.path.getctime))


def _run_dir(experiment_name: str, force_new: bool) -> Path:
    """
    Determine the log directory for this run.

    force_new=True  → always create a new run_XX folder (training fresh)
    force_new=False → reuse the latest run_XX folder (loading / eval)
    """
    base = ROOT / "logs" / experiment_name
    base.mkdir(parents=True, exist_ok=True)

    existing = sorted(base.glob("run_*"))
    if force_new or not existing:
        run_number = len(existing) + 1
    else:
        run_number = len(existing)          # reuse latest

    run_dir = base / f"run_{run_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Custom logging hook — writes reward components + termination counts
# ---------------------------------------------------------------------------

class RewardTermLogger:
    """
    Attached to the runner's TensorBoard writer.
    Call .step() once per RSL-RL iteration.

    Reads reward episode sums and termination info directly from the
    unwrapped CoopLiftEnv and logs them to TensorBoard.
    """

    def __init__(self, base_env: CoopLiftEnv, writer, log_interval: int):
        self.env          = base_env
        self.writer       = writer
        self.log_interval = log_interval
        self._iter        = 0

    def step(self, mean_reward: float, mean_episode_length: float) -> None:
        self._iter += 1
        if self._iter % self.log_interval != 0:
            return

        it = self._iter

        # ---- scalar: overall reward ----
        self.writer.add_scalar("Train/mean_reward",         mean_reward,          it)
        self.writer.add_scalar("Train/mean_episode_length", mean_episode_length,  it)

        # ---- reward components ----
        ep_sums = self.env._reward_manager.get_episode_sums()
        for name, tensor in ep_sums.items():
            # Average over all envs (per-episode mean)
            mean_val = tensor.mean().item()
            self.writer.add_scalar(f"Reward/{name}", mean_val, it)

        # ---- termination counts (fraction of envs terminated this iter) ----
        term_info = self.env._termination_manager.get_termination_info()
        for name, tensor in term_info.items():
            frac = tensor.float().mean().item()
            self.writer.add_scalar(f"Termination/{name}", frac, it)

        # ---- crate / drone debug scalars ----
        crate_z = self.env._crate.data.root_pos_w[:, 2].mean().item()
        self.writer.add_scalar("Debug/mean_crate_z", crate_z, it)

        drone0_z = self.env._drones["drone_0"].data.root_pos_w[:, 2].mean().item()
        self.writer.add_scalar("Debug/mean_drone0_z", drone0_z, it)


# ===========================================================================
# Environment factory
# ===========================================================================

def make_env(num_envs: int, episode_s: float, render_mode=None) -> FlattenedMARLWrapper:
    cfg = CoopLiftEnvCfg()
    cfg.scene.num_envs  = num_envs
    cfg.episode_length_s = episode_s
    base_env = CoopLiftEnv(cfg=cfg, render_mode=render_mode)
    print_env_info(base_env)
    return FlattenedMARLWrapper(base_env)


# ===========================================================================
# Main
# ===========================================================================

def main():
    register_envs()

    # ------------------------------------------------------------------
    # 1. Build env + RSL-RL wrapper
    # ------------------------------------------------------------------
    render_mode = "rgb_array" if args.mode == "eval" else None
    flat_env    = make_env(args.num_envs, args.episode_s, render_mode=render_mode)
    rsl_env     = RslRlVecEnvWrapper(flat_env)

    device = rsl_env.device
    print(f"\n[main] device       : {device}")
    print(f"[main] num_envs     : {args.num_envs}")
    print(f"[main] num_drones   : {NUM_DRONES}")
    print(f"[main] flat agents  : {flat_env.num_envs}")
    print(f"[main] obs_dim      : {flat_env.observation_space.shape}")
    print(f"[main] action_dim   : {flat_env.action_space.shape}\n")

    # ------------------------------------------------------------------
    # 2. Runner config
    # ------------------------------------------------------------------
    runner_cfg_obj = create_runner_cfg()
    agent_cfg      = (runner_cfg_obj.to_dict()
                      if hasattr(runner_cfg_obj, "to_dict")
                      else vars(runner_cfg_obj))

    experiment_name = agent_cfg["experiment_name"]
    force_new       = not (args.load or args.mode == "eval")
    log_dir         = _run_dir(experiment_name, force_new=force_new)

    # Inject dynamic run name so RSL-RL saves correctly
    agent_cfg["run_name"]       = log_dir.name
    agent_cfg["log_root_path"]  = str(log_dir)
    print(f"[main] log dir: {log_dir}")

    # ------------------------------------------------------------------
    # 3. Initialise runner
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(
        env=rsl_env,
        train_cfg=agent_cfg,
        log_dir=log_dir,
        device=device,
    )

    # ------------------------------------------------------------------
    # 4. Attach custom reward / termination logger
    # ------------------------------------------------------------------
    base_env = flat_env.env   # unwrap FlattenedMARLWrapper → CoopLiftEnv

    rew_term_logger = None
    if hasattr(runner, "writer") and runner.writer is not None:
        rew_term_logger = RewardTermLogger(
            base_env=base_env,
            writer=runner.writer,
            log_interval=args.log_interval,
        )
    else:
        print("[main] WARNING: runner has no TensorBoard writer — "
              "reward/termination logging disabled.")

    # ------------------------------------------------------------------
    # 5. Load checkpoint (if requested)
    # ------------------------------------------------------------------
    resume_path = None

    if args.checkpoint:
        resume_path = args.checkpoint
    elif args.load or args.mode == "eval":
        # Search in the current log dir first, then any previous run
        for run in sorted((ROOT / "logs" / experiment_name).glob("run_*"),
                          reverse=True):
            found = _latest_checkpoint(run)
            if found:
                resume_path = found
                break

    if resume_path:
        if not Path(resume_path).exists():
            print(f"[main] ERROR: checkpoint not found: {resume_path}")
            sys.exit(1)
        print(f"[main] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
    elif args.mode == "eval":
        print("[main] WARNING: eval mode but no checkpoint found — "
              "running with untrained policy.")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    if args.mode == "train":

        print(f"\n[main] Starting training for {agent_cfg['max_iterations']} iterations …\n")

        # Monkey-patch the runner's log() to inject our extra scalars.
        # RSL-RL calls self.log(locs) at the end of each learn() iteration
        # where locs is a dict with "mean_reward", "mean_episode_length", etc.
        _original_log = runner.log

        def _patched_log(locs: dict, width: int = 80, pad: int = 35) -> None:
            _original_log(locs, width, pad)
            if rew_term_logger is not None:
                rew_term_logger.step(
                    mean_reward=locs.get("mean_reward", 0.0),
                    mean_episode_length=locs.get("mean_episode_length", 0.0),
                )

        runner.log = _patched_log

        runner.learn(
            num_learning_iterations=agent_cfg["max_iterations"],
            init_at_random_ep_len=True,
        )

        print("\n[main] Training complete.")

    # ------------------------------------------------------------------
    # 7. Evaluation / video recording loop
    # ------------------------------------------------------------------
    elif args.mode == "eval":
        print(f"\n[main] Starting eval for {args.eval_steps} steps …\n")

        policy = runner.get_inference_policy(device=device)
        obs, _ = rsl_env.reset()

        # ---- video setup ----
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

        video_dir  = ROOT / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"coop_lift_{log_dir.name}.mp4"

        writer = imageio.get_writer(str(video_path), fps=30)
        print(f"[eval] Recording to: {video_path}")

        # ---- viewport capture setup ----
        try:
            import ctypes
            from omni.kit.viewport.utility import (
                get_active_viewport,
                capture_viewport_to_buffer,
            )
            viewport_api = get_active_viewport()
            capture_available = viewport_api is not None
        except Exception as e:
            print(f"[eval] Viewport capture unavailable: {e}")
            capture_available = False

        last_frame = [None]

        if capture_available:
            pyapi = ctypes.pythonapi
            pyapi.PyCapsule_GetPointer.restype  = ctypes.c_void_p
            pyapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

            def _on_capture(buffer, size, width, height, fmt):
                try:
                    ptr       = pyapi.PyCapsule_GetPointer(buffer, None)
                    img_bytes = ctypes.string_at(ptr, size)
                    img       = np.frombuffer(img_bytes, dtype=np.uint8).reshape(height, width, 4)
                    last_frame[0] = img[:, :, :3]
                except Exception as ex:
                    pass  # silently skip bad frames

        # ---- per-step accumulators for eval stats ----
        total_reward    = torch.zeros(rsl_env.num_envs, device=device)
        episode_lengths = torch.zeros(rsl_env.num_envs, device=device)

        # Termination counters across eval window
        term_counts: dict[str, int] = {}

        # ---- eval loop ----
        with torch.no_grad():
            for step in range(args.eval_steps):

                actions = policy(obs)
                obs, rewards, dones, infos = rsl_env.step(actions)

                total_reward    += rewards
                episode_lengths += 1

                # --- termination tracking ---
                term_info = base_env._termination_manager.get_termination_info()
                for name, mask in term_info.items():
                    term_counts[name] = term_counts.get(name, 0) + int(mask.sum().item())

                # --- camera follow (env 0 drone 0) ---
                if capture_available:
                    try:
                        robot_pos = base_env._drones["drone_0"].data.root_pos_w[0, :3].cpu().numpy()
                        eye       = robot_pos + np.array([3.0, 2.0, 2.5])
                        base_env.viewport_camera_controller.update_view_location(
                            eye=eye, lookat=robot_pos
                        )
                    except Exception:
                        pass

                    capture_viewport_to_buffer(viewport_api, _on_capture)
                    if last_frame[0] is not None:
                        writer.append_data(last_frame[0])
                        last_frame[0] = None

                # --- console progress ---
                if step % 100 == 0:
                    crate_z  = base_env._crate.data.root_pos_w[:, 2].mean().item()
                    mean_rew = total_reward.mean().item() / max(step + 1, 1)
                    print(f"  [eval step {step:4d}] "
                          f"mean_cumulative_reward={mean_rew:+.3f}  "
                          f"crate_z={crate_z:.3f} m")

        # ---- save video ----
        writer.close()
        print(f"\n[eval] Video saved: {video_path}")

        # ---- print eval summary ----
        print(f"\n{'='*55}")
        print("  Eval Summary")
        print(f"{'='*55}")
        print(f"  Steps           : {args.eval_steps}")
        print(f"  Mean tot reward : {total_reward.mean().item():+.3f}")
        print(f"  Min tot reward  : {total_reward.min().item():+.3f}")
        print(f"  Max tot reward  : {total_reward.max().item():+.3f}")
        print(f"\n  Termination counts (total across all envs & steps):")
        for name, count in term_counts.items():
            pct = 100.0 * count / (args.eval_steps * args.num_envs * NUM_DRONES + 1e-9)
            print(f"    {name:<30s}: {count:5d}  ({pct:.2f} %)")
        print(f"\n  Reward component episode sums (mean over envs):")
        ep_sums = base_env._reward_manager.get_episode_sums()
        for name, tensor in ep_sums.items():
            print(f"    {name:<25s}: {tensor.mean().item():+.4f}")
        print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # 8. Cleanup
    # ------------------------------------------------------------------
    rsl_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()