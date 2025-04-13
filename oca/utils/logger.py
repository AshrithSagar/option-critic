"""
oca/utils/logger.py \n
Logging utilities
"""

import logging
import os
import time
from typing import Literal

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    def __init__(self, logdir: str, run_name: str):
        self.log_name = os.path.join(
            logdir, run_name.replace(" ", "_").replace("/", "_")
        )
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0

        os.makedirs(self.log_name, exist_ok=True)
        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + "/logger.log"),
            ],
            datefmt="%Y/%m/%d %I:%M:%S %p",
        )

    @property
    def time_diff(self, format: Literal["h", "m", "s"] = "h") -> float:
        diff = time.time() - self.start_time
        if format == "h":
            return diff / 60 / 60
        elif format == "m":
            return diff / 60
        elif format == "s":
            return diff
        raise ValueError("Invalid format")


class RegularLogger(BaseLogger):
    def __init__(self, logdir, run_name):
        super().__init__(logdir, run_name)

    def log_episode(self, steps, reward):
        self.n_eps += 1
        logging.info(
            f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} "
            f"| hours={self.time_diff:.3f}"
        )
        self.writer.add_scalar(
            tag="episodic_rewards", scalar_value=reward, global_step=self.n_eps
        )


class OptionsLogger(BaseLogger):
    def __init__(self, logdir, run_name):
        super().__init__(logdir, run_name)

    def log_episode(self, steps, reward, option_lengths, ep_steps, epsilon):
        self.n_eps += 1
        logging.info(
            f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "
            f"| hours={self.time_diff:.3f} | epsilon={epsilon:.3f}"
        )
        self.writer.add_scalar(
            tag="episodic_rewards", scalar_value=reward, global_step=self.n_eps
        )
        self.writer.add_scalar(
            tag="episode_lengths", scalar_value=ep_steps, global_step=self.n_eps
        )

        # Keep track of options statistics
        for option, lens in option_lengths.items():
            # Need better statistics for this one, point average is terrible in this case
            self.writer.add_scalar(
                tag=f"option_{option}_avg_length",
                scalar_value=np.mean(lens) if len(lens) > 0 else 0,
                global_step=self.n_eps,
            )
            self.writer.add_scalar(
                tag=f"option_{option}_active",
                scalar_value=sum(lens) / ep_steps,
                global_step=self.n_eps,
            )

    def log_data(self, step, actor_loss, critic_loss, entropy, epsilon):
        if actor_loss:
            self.writer.add_scalar(
                tag="actor_loss", scalar_value=actor_loss.item(), global_step=step
            )
        if critic_loss:
            self.writer.add_scalar(
                tag="critic_loss", scalar_value=critic_loss.item(), global_step=step
            )
        self.writer.add_scalar(
            tag="policy_entropy", scalar_value=entropy, global_step=step
        )
        self.writer.add_scalar(tag="epsilon", scalar_value=epsilon, global_step=step)


if __name__ == "__main__":
    logger = OptionsLogger(logdir="runs/", run_name="test_model-test_env")
    steps = 200
    reward = 5
    option_lengths = {opt: np.random.randint(0, 5, size=(5)) for opt in range(5)}
    ep_steps = 50
    epsilon = 0.99
    logger.log_episode(steps, reward, option_lengths, ep_steps, epsilon)
