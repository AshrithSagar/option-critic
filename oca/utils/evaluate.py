"""
oca/evaluate.py \n
Evaluate a trained agent
"""

import numpy as np
import torch

from ..agents.utils import get_agent_meth
from ..envs.utils import make_env
from .config import ConfigEvalProto


def main(args: ConfigEvalProto):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env, is_atari = make_env(args.env, render_mode=args.render_mode)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    evaluate = get_agent_meth("evaluate", args.agent)
    evaluate(args, env, device=device, is_atari=is_atari)
