"""
oca/utils/experience_replay.py \n
Experience replay buffer
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(
            *self.rng.sample(self.buffer, batch_size)
        )
        return np.stack(obs), option, reward, np.stack(next_obs), done

    def __len__(self):
        return len(self.buffer)
