"""
oca/agents/sarsa.py \n
SARSA agent
"""

import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor

from ..envs.fourrooms import FourRoomsEnv
from ..envs.utils import OneHotWrapper
from ..utils.config import ConfigRunProto
from ..utils.constants import models_dir
from ..utils.logger import RegularLogger


class SARSAAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.01,
        temperature: float = 0.001,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.gamma = gamma

        # Q-network approximator
        self.q_net = nn.Linear(state_dim, action_dim, bias=False)
        nn.init.zeros_(self.q_net.weight)  # Initialize all weights to zero
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=lr)

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


def run(args: ConfigRunProto, env: gym.Env, **kwargs):
    state_dim: int = env.observation_space.shape[0]  # if one-hot state
    action_dim: int = env.action_space.n
    env = OneHotWrapper(env)

    agent = SARSAAgent(
        state_dim,
        action_dim,
        lr=args.learning_rate,
        temperature=args.temperature,
        gamma=args.gamma,
    )

    logger = RegularLogger(
        logdir=args.logdir,
        run_name=f"{SARSAAgent.__name__}-{args.env}-{args.exp_name}-{time.ctime()}",
    )

    ep = 0
    if args.switch_goal:
        assert isinstance(env.unwrapped, FourRoomsEnv)
        print(f"Current goal {env.unwrapped.goal}")
    while ep < args.max_steps_total:
        reward = agent.train_episode(env)

        if args.switch_goal and logger.n_eps == 1000:
            torch.save(
                {"model_params": agent.state_dict(), "goal_state": env.unwrapped.goal},
                f"{models_dir}/sarsa_seed={args.seed}_1k",
            )
            env.unwrapped.switch_goal()
            print(f"New goal {env.unwrapped.goal}")

        if args.switch_goal and logger.n_eps > 2000:
            torch.save(
                {"model_params": agent.state_dict(), "goal_state": env.unwrapped.goal},
                f"{models_dir}/sarsa_seed={args.seed}_2k",
            )
            break

        ep += 1
        env.render()

        logger.log_episode(ep, reward)


def evaluate(args: ConfigRunProto, env: gym.Env, **kwargs):
    raise NotImplementedError
