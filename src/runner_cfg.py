#!/usr/bin/env python3
"""
runner_cfg.py
RSL-RL OnPolicyRunner configuration for the cooperative crate-lift task.

Tuning notes
------------
- num_steps_per_env: 64 → ~64 * num_envs transitions per update.
  Increase to 128 if gradients are noisy early in training.
- max_iterations: 5000 is usually enough to see emergent cooperative lifting.
  Increase to 10 000 for a full run.
- entropy_coef: 0.005 (slightly higher than locomotion tasks) to keep
  exploration alive — the cooperative task has a sparse success signal.
- actor/critic hidden dims: 512/256/128 — lighter than the Steve walker
  because each agent's obs_dim is only ~26 features.
- clip_actions: keep at 1.0 since ActionManager already clamps thrust.
- obs_groups: both policy and critic see the same per-agent obs.  There is
  no privileged critic info here; add a "critic" group later if you add
  crate mass / rope tension as privileged state.
"""

from __future__ import annotations
from pathlib import Path

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def create_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return a fully configured RslRlOnPolicyRunnerCfg for CoopLift."""

    # ------------------------------------------------------------------
    # PPO algorithm hyper-parameters
    # ------------------------------------------------------------------
    algorithm_cfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        # Learning rate — moderate; scheduler will anneal if desired
        learning_rate=3e-4,
        # Number of SGD epochs per rollout batch
        num_learning_epochs=8,
        # Number of mini-batches per epoch
        # total_steps = num_envs * num_drones * num_steps_per_env
        # mini_batch_size = total_steps / num_mini_batches
        num_mini_batches=4,
        # Discount factor — 0.99 rewards long-horizon cooperative behaviour
        gamma=0.994,
        # GAE lambda
        lam=0.95,
        # Entropy coefficient — slightly elevated for sparse-ish lift task
        entropy_coef=0.005,
        # KL divergence target for adaptive lr
        desired_kl=0.01,
        # PPO clip range
        clip_param=0.2,
        # Normalise advantages inside each mini-batch
        normalize_advantage_per_mini_batch=True,
        # Value function loss weight
        value_loss_coef=0.025,
        # Gradient clipping
        max_grad_norm=1.0,
    )

    # ------------------------------------------------------------------
    # Actor-Critic network
    # ------------------------------------------------------------------
    policy_cfg = RslRlPpoActorCriticCfg(
        # Initial action noise std — moderate; log-parameterised std adapts
        init_noise_std=0.8,
        noise_std_type="log",
        # Running mean/std normalisation for both actor and critic inputs
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Network widths — lighter than locomotion, obs is only ~26-d
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        # ELU gives smooth gradients, works well with continuous control
        activation="elu",
    )

    # ------------------------------------------------------------------
    # Runner (top-level)
    # ------------------------------------------------------------------
    runner_cfg = RslRlOnPolicyRunnerCfg(
        # ---- identifiers ----
        experiment_name="coop_lift",
        run_name="run_01",           # overwritten dynamically in train.py

        # ---- sampling ----
        # Rollout horizon per environment per update
        num_steps_per_env=256,
        # Total gradient updates — increase for a full training run
        max_iterations=5000,

        # ---- observation groups ----
        # "policy" obs → actor; "critic" obs → critic value head.
        # Both share the same per-agent obs here (no privileged info).
        obs_groups={
            "policy": ["policy"],
            "critic": ["policy"],
        },

        # ---- action clipping ----
        # ActionManager already clamps; this is a safety net in RSL-RL
        clip_actions=1.0,

        # ---- logging & saving ----
        logger="tensorboard",
        save_interval=100,           # save .pt every N iterations

        # ---- sub-configs ----
        policy=policy_cfg,
        algorithm=algorithm_cfg,

        # ---- misc ----
        seed=42,
    )

    return runner_cfg