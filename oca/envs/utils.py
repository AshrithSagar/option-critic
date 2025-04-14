"""
oca/envs/utils.py \n
Utility functions for the environments
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformReward,
)
from numpy.typing import NDArray

from .fourrooms import FourRoomsEnv

gym.register_envs(ale_py)

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

def is_atari_env(env_id):
    try:
        env = gym.make(env_id, render_mode='human')
        # Check if 'ALE' is in the environment's metadata (typical for Atari)
        return 'ale' in env.spec.entry_point.lower()
    except Exception as e:
        print(f"Error loading environment {env_id}: {e}")
        return False
    
def make_env(env_name: str, **kwargs) -> Tuple[gym.Env, bool]:
    if env_name == "fourrooms":
        return FourRoomsEnv(), False

    env = gym.make(env_name, **kwargs)
    is_atari = is_atari_env(env_name)
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


# convert discrete state to one-hot vector in wrapper
class OneHotWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        self.env = env
        self.n = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        if hasattr(env, "goal"):
            self.goal = env.goal
        if hasattr(env, "switch_goal"):
            self.switch_goal = env.switch_goal

    def reset(self) -> Tuple[NDArray, dict]:
        s: NDArray
        s, *_ = self.env.reset()
        return self._one_hot(s), {}

    def step(self, a: int) -> Tuple[NDArray, float, bool, bool, dict]:
        s: NDArray
        s, r, done, _, info = self.env.step(a)
        return self._one_hot(s), r, done, _, info

    def _one_hot(self, s: NDArray) -> NDArray:
        vec = np.zeros(self.n, dtype=np.float32)
        scalar_s = s.item() if np.isscalar(s) else s[0]
        vec[int(scalar_s)] = 1.0
        return vec
