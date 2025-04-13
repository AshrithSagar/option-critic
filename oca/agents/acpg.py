"""
oca/agents/acpg.py \n
Actor-Critic agent with policy gradient
"""

import time
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.types import Number, Tensor

from ..envs.fourrooms import FourRoomsEnv
from ..envs.utils import OneHotWrapper
from ..utils.config import ConfigRunProto
from ..utils.constants import models_dir
from ..utils.logger import RegularLogger


class ActorCriticAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 0.001,
        lr_critic: float = 0.01,
        temperature: float = 0.001,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.gamma = gamma
        self.temperature = temperature
        # policy: linear softmax
        self.policy_net = nn.Linear(state_dim, action_dim, bias=False)
        nn.init.zeros_(self.policy_net.weight)  # Initialize all weights to zero
        # value: linear
        self.value_net = nn.Linear(state_dim, 1, bias=False)
        nn.init.zeros_(self.value_net.weight)  # Initialize all weights to zero
        # separate optimizers for actor and critic
        self.optimizer_actor = optim.RMSprop(self.policy_net.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.RMSprop(self.value_net.parameters(), lr=lr_critic)

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
        action = m.sample()
        return action.item(), m.log_prob(action)

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

        # Critic: compute TD target and TD error (advantage)
        value: Tensor = self.value_net(state_tensor).squeeze(0)
        next_value: Tensor = (
            torch.tensor(0.0)
            if done
            else self.value_net(next_state_tensor).squeeze(0).detach()
        )
        td_target = reward + self.gamma * next_value
        advantage = td_target - value

        # Critic update (MSE loss)
        critic_loss = advantage.pow(2)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor update (policy gradient)
        actor_loss = -log_prob * advantage.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> float:
        state: NDArray
        state, _ = env.reset()
        total_reward: float = 0.0
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


def run_acpg(args: ConfigRunProto, env: gym.Env, **kwargs):
    state_dim: int = env.observation_space.shape[0]  # if one-hot state
    action_dim: int = env.action_space.n
    env = OneHotWrapper(env)

    agent = ActorCriticAgent(
        state_dim,
        action_dim,
        lr_actor=args.learning_rate,
        lr_critic=args.learning_rate,
        temperature=args.temperature,
        gamma=args.gamma,
    )

    logger = RegularLogger(
        logdir=args.logdir,
        run_name=f"{ActorCriticAgent.__name__}-{args.env}-{args.exp_name}-{time.ctime()}",
    )

    ep = 0
    if args.switch_goal:
        env: FourRoomsEnv
        print(f"Current goal {env.goal}")
    while ep < args.max_steps_total:
        reward = agent.train_episode(env)

        if args.switch_goal and logger.n_eps == 1000:
            torch.save(
                {"model_params": agent.state_dict(), "goal_state": env.goal},
                f"{models_dir}/acpg_seed={args.seed}_1k",
            )
            env.switch_goal()
            print(f"New goal {env.goal}")

        if args.switch_goal and logger.n_eps > 2000:
            torch.save(
                {"model_params": agent.state_dict(), "goal_state": env.goal},
                f"{models_dir}/acpg_seed={args.seed}_2k",
            )
            break

        ep += 1
        env.render()

        logger.log_episode(ep, reward)
