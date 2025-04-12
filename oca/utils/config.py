"""
utils/config.py \n
Configuration utilities
"""

from typing import Dict, Optional, Protocol

import yaml


class ConfigProto(Protocol):
    env: str  # ROM to run
    optimal_eps: float  # Epsilon when playing optimally
    frame_skip: int  # Every how many frames to process
    learning_rate: float  # Learning rate
    gamma: float  # Discount rate
    epsilon_start: float  # Starting value for epsilon
    epsilon_min: float  # Minimum epsilon
    epsilon_decay: float  # Number of steps to minimum epsilon
    max_history: int  # Maximum number of steps stored in replay
    batch_size: int  # Batch size
    freeze_interval: int  # Interval between target freezes
    update_frequency: int  # Number of actions before each SGD update
    termination_reg: float  # Regularization to decrease termination prob
    entropy_reg: float  # Regularization to increase policy entropy
    num_options: int  # Number of options to create
    temp: float  # Action distribution softmax temperature param
    max_steps_ep: int  # Number of maximum steps per episode
    max_steps_total: int  # Number of maximum steps in total
    cuda: bool  # Enable CUDA training (recommended if possible)
    seed: int  # Random seed for numpy, torch, random.
    logdir: str  # Directory for logging statistics
    exp: Optional[str]  # Optional experiment name
    switch_goal: bool  # Switch goal after 2k eps


class ConfigDefaults:
    env = "CartPole-v1"
    optimal_eps = 0.05
    frame_skip = 4
    learning_rate = 0.0005
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 20000
    max_history = 10000
    batch_size = 32
    freeze_interval = 200
    update_frequency = 4
    termination_reg = 0.01
    entropy_reg = 0.01
    num_options = 2
    temp = 1.0
    max_steps_ep = 18000
    max_steps_total = 4e6
    cuda = True
    seed = 0
    logdir = "runs"
    exp = None
    switch_goal = False


def load_config(config_path: Optional[str] = None) -> ConfigProto:
    config = ConfigDefaults()
    if config_path:
        # Override default config with custom config if provided
        with open(config_path, "r") as file:
            config_dict: Dict = yaml.safe_load(file)
            for key, value in config_dict.items():
                setattr(config, key, value)
    return config
