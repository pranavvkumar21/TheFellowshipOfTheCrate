#!/usr/bin/env python3
"""
registration.py
Registers the CoopLift flattened MARL environment with Gymnasium.

The env exposed to RSL-RL is the FlattenedMARLWrapper version so that
OnPolicyRunner sees a standard (num_envs * NUM_DRONES, obs_dim) VecEnv.
"""

from __future__ import annotations
import gymnasium as gym


def register_envs() -> None:
    """
    Call this once before gym.make().

    Registered IDs
    --------------
    CoopLift-v0   : raw DirectMARLEnv (useful for debugging)
    CoopLift-Flat-v0 : FlattenedMARLWrapper — what the RSL-RL runner uses
    """

    # ------------------------------------------------------------------
    # 1. Raw DirectMARLEnv — handy for direct inspection
    # ------------------------------------------------------------------
    if "CoopLift-v0" not in gym.envs.registry:
        gym.register(
            id="CoopLift-v0",
            entry_point="quadcopter_lift_env:CoopLiftEnv",
            disable_env_checker=True,
            kwargs={"cfg": None},   # caller must pass cfg= at make-time
        )

    # ------------------------------------------------------------------
    # 2. Flattened wrapper — used by the training runner
    # ------------------------------------------------------------------
    if "CoopLift-Flat-v0" not in gym.envs.registry:
        gym.register(
            id="CoopLift-Flat-v0",
            entry_point="registration:_make_flat_env",   # factory below
            disable_env_checker=True,
            kwargs={"cfg": None},
        )

    print("[registration] CoopLift-v0 and CoopLift-Flat-v0 registered.")


# ---------------------------------------------------------------------------
# Factory used by the Gymnasium entry_point string above
# ---------------------------------------------------------------------------

def _make_flat_env(cfg, render_mode=None, **kwargs):
    """
    Instantiates CoopLiftEnv then wraps it in FlattenedMARLWrapper.
    cfg must be a CoopLiftEnvCfg instance supplied by the caller.
    """
    from quadcopter_lift_env import CoopLiftEnv
    from wrapper import FlattenedMARLWrapper

    if cfg is None:
        from quadcopter_lift_env_cfg import CoopLiftEnvCfg
        cfg = CoopLiftEnvCfg()

    base_env = CoopLiftEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    return FlattenedMARLWrapper(base_env)