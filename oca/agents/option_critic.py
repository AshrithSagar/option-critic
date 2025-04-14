"""
oca/utils/option_critic.py \n
Option Critic Architecture
"""

import time
from copy import deepcopy
from math import exp

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch.types import Tensor

from ..envs.fourrooms import FourRoomsEnv
from ..envs.utils import to_tensor
from ..utils.config import ConfigRunProto
from ..utils.constants import models_dir
from ..utils.experience_replay import ReplayBuffer
from ..utils.logger import OptionsLogger


class OptionCriticBase(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_actions,
        num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_options = num_options
        self.feature_dim = feature_dim
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        # Shared heads
        self.Q = nn.Linear(feature_dim, num_options)
        self.terminations = nn.Linear(feature_dim, num_options)
        self.options_W = nn.Parameter(
            torch.zeros(num_options, feature_dim, num_actions)
        )
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_Q(self, state):
        return self.Q(state)

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_action(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(
                -self.num_steps / self.eps_decay
            )
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        return self.features(obs)


class OptionCriticConv(OptionCriticBase):
    """
    For Visual Input (CNN)
    """

    def __init__(self, in_features, num_actions, num_options, **kwargs):
        self.magic_number = 7 * 7 * 64  # Output of CNN
        super().__init__(
            feature_dim=512, num_actions=num_actions, num_options=num_options, **kwargs
        )

        self.features = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU(),
        )


class OptionCriticFeatures(OptionCriticBase):
    """
    For Vector Input (MLP)
    """

    def __init__(self, in_features, num_actions, num_options, **kwargs):
        super().__init__(
            feature_dim=64, num_actions=num_actions, num_options=num_options, **kwargs
        )

        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )


def critic_loss_fn(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options = torch.LongTensor(options).to(model.device)
    rewards = torch.FloatTensor(rewards).to(model.device)
    masks = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q = model.get_Q(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime = model_prime.get_Q(next_states_prime)  # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * (
        (1 - next_options_term_prob) * next_Q_prime[batch_idx, options]
        + next_options_term_prob * next_Q_prime.max(dim=-1)[0]
    )

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err


def actor_loss_fn(
    obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args
):
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * (
        (1 - next_option_term_prob) * next_Q_prime[option]
        + next_option_term_prob * next_Q_prime.max(dim=-1)[0]
    )

    # The termination loss
    termination_loss = (
        option_term_prob
        * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg)
        * (1 - done)
    )

    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss


def _get_model(args: ConfigRunProto, env: gym.Env, **kwargs):
    option_critic = OptionCriticConv if kwargs["is_atari"] else OptionCriticFeatures
    model = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temperature,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=kwargs["device"],
    )
    return model


def run(args: ConfigRunProto, env: gym.Env, **kwargs):
    model = _get_model(args, env, **kwargs)

    # Create a prime network for more stable Q values
    model_prime = deepcopy(model)

    optim = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = OptionsLogger(
        logdir=args.logdir,
        run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp_name}-{time.ctime()}",
    )

    steps = 0
    if args.switch_goal:
        assert isinstance(env.unwrapped, FourRoomsEnv)
        print(f"Current goal {env.unwrapped.goal}")
    while steps < args.max_steps_total:
        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs, _ = env.reset()
        state = model.get_state(to_tensor(obs))
        greedy_option = model.greedy_option(state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal and logger.n_eps == 1000:
            torch.save(
                {"model_params": model.state_dict(), "goal_state": env.unwrapped.goal},
                f"{models_dir}/oca_seed={args.seed}_1k",
            )
            env.unwrapped.switch_goal()
            print(f"New goal {env.unwrapped.goal}")

        if args.switch_goal and logger.n_eps > 2000:
            torch.save(
                {"model_params": model.state_dict(), "goal_state": env.unwrapped.goal},
                f"{models_dir}/oca_seed={args.seed}_2k",
            )
            break

        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = model.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = (
                    np.random.choice(args.num_options)
                    if np.random.rand() < epsilon
                    else greedy_option
                )
                curr_op_len = 0

            action, logp, entropy = model.get_action(state, current_option)

            next_obs, reward, done, _, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(
                    obs,
                    current_option,
                    logp,
                    entropy,
                    reward,
                    done,
                    next_obs,
                    model,
                    model_prime,
                    args,
                )
                loss: Tensor = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(model, model_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    model_prime.load_state_dict(model.state_dict())

            state = model.get_state(to_tensor(next_obs))
            (
                option_termination,
                greedy_option,
            ) = model.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            env.render()

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)

    env.close()


def evaluate(args: ConfigRunProto, env: gym.Env, **kwargs):
    model = _get_model(args, env, **kwargs)
    checkpoint = torch.load(args.model_path, map_location=model.device)
    model.load_state_dict(checkpoint["model_params"])
    model.eval()

    total_reward = 0
    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        state = model.get_state(
            torch.tensor(obs, dtype=torch.float32, device=model.device)
        )
        term, trunc = False, False

        ep_steps = 0
        while not (term or trunc) and ep_steps < args.max_steps_ep:
            env.render()
            option = model.greedy_option(state)
            action, _, _ = model.get_action(state, option)
            next_obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            state = model.get_state(
                torch.tensor(next_obs, dtype=torch.float32, device=model.device)
            )
            ep_steps += 1

        print(
            f"Episode {ep + 1} | Total Reward = {total_reward} | Reward = {reward} | Steps = {ep_steps}"
        )

    env.close()
