"""
oca/run.py \n
Option-Critic Architecture
"""

import numpy as np
import torch

from ..agents.utils import get_agent_run
from ..envs.utils import make_env
from .config import ConfigRunProto


def main(args: ConfigRunProto):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env, is_atari = make_env(args.env, render_mode="human")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    run = get_agent_run(args.agent)
    run(args, env, device=device, is_atari=is_atari)
