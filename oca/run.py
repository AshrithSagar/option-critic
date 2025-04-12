"""
oca/run.py \n
Option-Critic Architecture
"""

import time
from copy import deepcopy

import numpy as np
import torch

from .envs.utils import make_env, to_tensor
from .utils.cli import load_config
from .utils.constants import models_dir
from .utils.experience_replay import ReplayBuffer
from .utils.logger import Logger
from .utils.option_critic import OptionCriticConv, OptionCriticFeatures
from .utils.option_critic import actor_loss as actor_loss_fn
from .utils.option_critic import critic_loss as critic_loss_fn


def main():
    args = load_config(verbose=True)
    env, is_atari = make_env(args.env, render_mode="human")
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    oca = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device,
    )
    # Create a prime network for more stable Q values
    oca_prime = deepcopy(oca)

    optim = torch.optim.RMSprop(oca.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(
        logdir=args.logdir,
        run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}",
    )

    steps = 0
    if args.switch_goal:
        print(f"Current goal {env.goal}")
    while steps < args.max_steps_total:
        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs, _ = env.reset()
        state = oca.get_state(to_tensor(obs))
        greedy_option = oca.greedy_option(state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal and logger.n_eps == 1000:
            torch.save(
                {"model_params": oca.state_dict(), "goal_state": env.goal},
                f"{models_dir}/oca_seed={args.seed}_1k",
            )
            env.switch_goal()
            print(f"New goal {env.goal}")

        if args.switch_goal and logger.n_eps > 2000:
            torch.save(
                {"model_params": oca.state_dict(), "goal_state": env.goal},
                f"{models_dir}/oca_seed={args.seed}_2k",
            )
            break

        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = oca.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = (
                    np.random.choice(args.num_options)
                    if np.random.rand() < epsilon
                    else greedy_option
                )
                curr_op_len = 0

            action, logp, entropy = oca.get_action(state, current_option)

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
                    oca,
                    oca_prime,
                    args,
                )
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(oca, oca_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    oca_prime.load_state_dict(oca.state_dict())

            state = oca.get_state(to_tensor(next_obs))
            (
                option_termination,
                greedy_option,
            ) = oca.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        env.render()

        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)


if __name__ == "__main__":
    main()
