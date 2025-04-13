"""
oca/envs/utils.py \n
Utility functions for the environments
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformReward,
)

from .fourrooms import FourRoomsEnv


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStackObservation):
    def __init__(self, env, k):
        FrameStackObservation.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def make_env(env_name: str, **kwargs) -> Tuple[gym.Env, bool]:
    if env_name == "fourrooms":
        return FourRoomsEnv(), False

    env = gym.make(env_name, **kwargs)
    is_atari = hasattr(gym.envs, "atari") and isinstance(
        env.unwrapped, gym.envs.atari.atari_env.AtariEnv
    )
    if is_atari:
        env = AtariPreprocessing(
            env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True
        )
        env = TransformReward(env, lambda r: np.clip(r, -1, 1))
        env = FrameStack(env, 4)
    return env, is_atari


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs
