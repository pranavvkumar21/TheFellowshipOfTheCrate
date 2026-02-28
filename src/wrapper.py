# wrapper.py

from __future__ import annotations
import numpy as np
import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from quadcopter_lift_env_cfg import NUM_DRONES


class FlattenedMARLWrapper(DirectRLEnv):
    """
    Flattens CoopLiftEnv (DirectMARLEnv) into a single-agent VecEnv that
    RslRlVecEnvWrapper accepts.

    Every attribute that RslRlVecEnvWrapper.__init__ and its helpers touch
    is explicitly provided here, derived by reading the full wrapper source:

        num_envs                 — flat n*A count
        device                   — passthrough
        max_episode_length       — passthrough
        single_action_space      — Box(act_dim,)   ← flatdim + _modify_action_space
        action_space             — Box(n*A, act_dim) ← _modify_action_space writes this
        observation_space        — Box(n*A, obs_dim)
        episode_length_buf       — property WITH setter (RSL-RL writes it for random init)
        cfg.is_finite_horizon    — shim
        cfg.num_actions          — shim
        render_mode              — passthrough
        unwrapped                — returns self
        _get_observations()      — returns {"policy": flat_obs}
        seed()                   — passthrough
        reset()                  — returns ({"policy": flat_obs}, extras)
        step()                   — returns ({"policy": flat_obs}, rew, term, trunc, extras)
    """

    # ------------------------------------------------------------------
    # Bypass DirectRLEnv.__init__ — it would launch a second simulation
    # ------------------------------------------------------------------

    def __new__(cls, env):
        return object.__new__(cls)

    def __init__(self, env):
        # All assignments go to private names to avoid clobbering
        # DirectRLEnv's read-only property descriptors.
        self._base_env   = env
        self._num_envs   = env.num_envs * NUM_DRONES
        self._device     = env.device
        self._sim        = env.sim
        self._scene      = env.scene
        self._render_mode = getattr(env, "render_mode", None)

        # Obs / action sizes
        self.obs_dim     = env._obs_manager.obs_dim
        self.action_dim  = env._action_manager.action_dim

        # RSL-RL shorthand (read by OnPolicyRunner directly)
        self.num_obs     = self.obs_dim
        self.num_actions = self.action_dim

        # single_action_space — RslRlVecEnvWrapper calls gym.spaces.flatdim()
        # on this, then _modify_action_space() overwrites it with clipped version.
        self.single_action_space = gym.spaces.Box(
            low=-np.ones(self.action_dim, dtype=np.float32),
            high= np.ones(self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # action_space — batch of single_action_space, also overwritten by
        # _modify_action_space() if clip_actions is set.
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self._num_envs
        )

        # observation_space
        inf = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-inf, high=inf, dtype=np.float32
        )


        observation_spaces = {"policy": self.observation_space}

        # cfg shim — RslRlVecEnvWrapper reads:
        #   cfg.is_finite_horizon  (step: controls time_outs key)
        #   cfg.num_actions        (some runner versions)
        self.cfg = type("_Cfg", (), {
            "is_finite_horizon": True,   # coop-lift episodes have a hard timeout
            "num_actions":       self.action_dim,
            "observation_spaces": observation_spaces,
        })()

        # episode_length_buf backing store — needs a setter (RSL-RL writes it)
        self._episode_length_buf = self._base_env.episode_length_buf.repeat_interleave(NUM_DRONES)

    # ------------------------------------------------------------------
    # Override every DirectRLEnv read-only property
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        """Return self so isinstance(env.unwrapped, DirectRLEnv) passes."""
        return self

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def sim(self):
        return self._sim

    @property
    def scene(self):
        return self._scene

    @property
    def render_mode(self):
        return self._render_mode

    @property
    def max_episode_length(self) -> int:
        return self._base_env.max_episode_length

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """(n*A,) — RSL-RL reads AND writes this for random episode-length init."""
        buf = self._base_env.episode_length_buf   # (n,)
        return buf.unsqueeze(1).expand(-1, NUM_DRONES).reshape(self._num_envs)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """
        RSL-RL writes a (n*A,) tensor here during random init.
        We fold it back to (n,) by taking the first agent's slice per env,
        then write it into the base env.
        """
        # value shape: (n*A,) — take every NUM_DRONES-th element (agent 0)
        per_env = value.reshape(self._base_env.num_envs, NUM_DRONES)[:, 0]
        self._base_env.episode_length_buf = per_env

    # Convenience alias for any code that does flat_env.env
    @property
    def env(self):
        return self._base_env

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        """
        RslRlVecEnvWrapper.reset() does:
            obs_dict, extras = self.env.reset()
            return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

        So obs_dict must be a plain dict keyed by obs group name.
        RSL-RL uses the "policy" key by default.
        """
        obs_dict, info = self._base_env.reset()
        flat_obs = self._flatten_obs(obs_dict)
        obs_out  = {"policy": flat_obs}
        extras   = {"observations": obs_out}
        extras.update(info)
        return obs_out, extras

    def step(self, actions: torch.Tensor):
        """
        RslRlVecEnvWrapper.step() does:
            obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
            dones = (terminated | truncated).to(dtype=torch.long)
            if not cfg.is_finite_horizon:
                extras["time_outs"] = truncated
            return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
        """
        action_dict = self._unflatten_actions(actions)

        obs_dict, reward_dict, terminated_dict, truncated_dict, info = \
            self._base_env.step(action_dict)

        flat_obs   = self._flatten_obs(obs_dict)
        obs_out    = {"policy": flat_obs}
        extras     = {"observations": obs_out}
        extras.update(info)

        return (
            obs_out,
            self._flatten_rewards(reward_dict),
            self._flatten_dones(terminated_dict),
            self._flatten_dones(truncated_dict),
            extras,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Called by RslRlVecEnvWrapper.get_observations() when the env has no
        observation_manager (our case).
        Must return a plain dict keyed by obs group.
        """
        obs_dict = self._base_env._obs_manager.compute()
        flat_obs = self._flatten_obs(obs_dict)
        return {"policy": flat_obs}

    def get_observations(self):
        """Direct call path (used by some runner versions)."""
        obs = self._get_observations()
        return obs, {"observations": obs}

    def seed(self, seed: int = -1) -> int:
        if hasattr(self._base_env, "seed"):
            return self._base_env.seed(seed)
        return seed

    # ------------------------------------------------------------------
    # Flatten / unflatten helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """{"drone_i": (n, D)} → (n*A, D)"""
        stacked = torch.stack(
            [obs_dict[f"drone_{i}"] for i in range(NUM_DRONES)],
            dim=1,
        )   # (n, A, D)
        return stacked.reshape(self._num_envs, self.obs_dim)

    def _unflatten_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """(n*A, act_dim) → {"drone_i": (n, act_dim)}"""
        n      = self._base_env.num_envs
        unflat = actions.reshape(n, NUM_DRONES, self.action_dim)
        return {f"drone_{i}": unflat[:, i, :] for i in range(NUM_DRONES)}

    def _flatten_rewards(self, reward_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """{"drone_i": (n,)} → (n*A,)"""
        stacked = torch.stack(
            [reward_dict[f"drone_{i}"] for i in range(NUM_DRONES)],
            dim=1,
        )   # (n, A)
        return stacked.reshape(self._num_envs)

    def _flatten_dones(self, signal_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """{"drone_i": (n,) bool} → (n*A,) bool — must be bool for bitwise_or in RslRlVecEnvWrapper"""
        done = signal_dict["drone_0"].bool()   # (n,)
        return done.unsqueeze(1).expand(-1, NUM_DRONES).reshape(self._num_envs)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def close(self):
        self._base_env.close()

    def render(self, *args, **kwargs):
        return self._base_env.render(*args, **kwargs)