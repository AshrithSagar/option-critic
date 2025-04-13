"""
oca/agents/sarsa.py \n
SARSA agent
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor

from oca.envs.fourrooms import FourRoomsEnv


class SARSAAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.01,
        temperature: float = 0.001,
        gamma: float = 0.99,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.gamma = gamma

        # Q-network approximator
        self.q_net = nn.Linear(state_dim, action_dim, bias=False)
        nn.init.zeros_(self.q_net.weight)  # Initialize all weights to zero
        self.optimizer = optim.SGD(self.q_net.parameters(), lr=lr)

    def get_q_values(self, state: NDArray) -> NDArray:
        # state: numpy array shape (state_dim,)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # (1, state_dim)
        with torch.no_grad():
            q_vals = self.q_net(state_tensor).squeeze(0)  # (action_dim,)
        return q_vals.numpy()

    def select_action(self, state: NDArray) -> Tuple[int, NDArray]:
        q_vals = self.get_q_values(state)

        # Boltzmann exploration
        prefs = q_vals / self.temperature
        max_pref = np.max(prefs)
        exp_prefs = np.exp(prefs - max_pref)
        probs: NDArray = exp_prefs / np.sum(exp_prefs)
        action = np.random.choice(self.action_dim, p=probs)

        return action, probs

    def update(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        next_action: int,
        done: bool,
    ):
        # Compute SARSA(0) target
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # (1, state_dim)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

        q_sa: Tensor = self.q_net(state_tensor)[0, action]
        with torch.no_grad():
            q_snext_anext = (
                0.0 if done else self.q_net(next_state_tensor)[0, next_action]
            )
            target: Tensor = reward + self.gamma * q_snext_anext

        # Loss and gradient step
        loss: Tensor = (q_sa - target) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> float:
        state: NDArray
        state, _ = env.reset()
        action, _ = self.select_action(state)
        total_reward: float = 0.0
        for t in range(max_steps):
            next_state: NDArray
            next_state, reward, done, _, _ = env.step(action)
            next_action, _ = self.select_action(next_state)
            self.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward
            if done:
                break
        return total_reward


# convert discrete state to one-hot vector in wrapper
class OneHotWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        self.env = env
        self.n = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space

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


def run_sarsa():
    env = FourRoomsEnv()
    state_dim: int = env.observation_space.shape[0]  # if one-hot state
    action_dim: int = env.action_space.n
    env = OneHotWrapper(env)

    agent = SARSAAgent(state_dim, action_dim, lr=0.01, temperature=0.001, gamma=0.99)

    num_episodes = 1000
    rewards = []
    for ep in range(num_episodes):
        R = agent.train_episode(env)
        rewards.append(R)
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, Reward: {np.mean(rewards[-100:]):.2f}")


if __name__ == "__main__":
    run_sarsa()
