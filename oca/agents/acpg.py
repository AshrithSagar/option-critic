"""
oca/agents/acpg.py \n
Actor-Critic agent with policy gradient
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.types import Number, Tensor

from oca.envs.fourrooms import FourRoomsEnv


class ActorCriticAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.01,
        temperature: float = 0.001,
        gamma: float = 0.99,
    ):
        self.gamma = gamma
        self.temperature = temperature
        # policy: linear softmax
        self.policy_net = nn.Linear(state_dim, action_dim, bias=False)
        nn.init.zeros_(self.policy_net.weight)  # Initialize all weights to zero
        # value: linear
        self.value_net = nn.Linear(state_dim, 1, bias=False)
        nn.init.zeros_(self.value_net.weight)  # Initialize all weights to zero
        self.optimizer = optim.SGD(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

    def get_policy(self, state: NDArray) -> NDArray:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        prefs = self.policy_net(state_tensor).squeeze(0) / self.temperature
        max_pref = prefs.max()
        exp_prefs = torch.exp(prefs - max_pref)
        probs = exp_prefs / exp_prefs.sum()
        return probs

    def select_action(self, state: NDArray) -> Tuple[Number, Tensor]:
        probs = self.get_policy(state)
        m = Categorical(probs)
        action = m.sample().item()
        return action, m.log_prob(torch.tensor(action))

    def update(
        self,
        state: NDArray,
        reward: float,
        next_state: NDArray,
        done: bool,
        log_prob: Tensor,
    ):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

        value: Tensor = self.value_net(state_tensor).squeeze(0)
        next_value: Tensor = (
            0.0 if done else self.value_net(next_state_tensor).squeeze(0).detach()
        )
        td_target = reward + self.gamma * next_value
        td_error = td_target - value

        # Loss: critic MSE + actor policy gradient
        value_loss: Tensor = td_error.pow(2)
        policy_loss: Tensor = -log_prob * td_error.detach()
        loss: Tensor = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> float:
        state: NDArray
        state, _ = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            action, log_prob = self.select_action(state)
            next_state: NDArray
            next_state, reward, done, _, _ = env.step(action)
            self.update(state, reward, next_state, done, log_prob)
            state = next_state
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


def run_acpg():
    env = FourRoomsEnv()
    state_dim: int = env.observation_space.shape[0]  # if one-hot state
    action_dim: int = env.action_space.n
    env = OneHotWrapper(env)

    agent = ActorCriticAgent(
        state_dim, action_dim, lr=0.01, temperature=0.001, gamma=0.99
    )

    num_episodes = 1000
    rewards = []
    for ep in range(num_episodes):
        R = agent.train_episode(env)
        rewards.append(R)
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, Reward: {np.mean(rewards[-100:]):.2f}")


if __name__ == "__main__":
    run_acpg()
