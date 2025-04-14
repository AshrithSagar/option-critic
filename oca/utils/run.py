"""
oca/run.py \n
Option-Critic Architecture
"""

import numpy as np
import torch

from ..agents.utils import get_agent_meth
from ..envs.utils import make_env
from .config import ConfigRunProto


def main(args: ConfigRunProto):
    if args.eval:
        args.render_mode = args.render_mode or "human"
        assert args.model_path, "Model path must be specified for evaluation"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env, is_atari = make_env(args.env, render_mode=args.render_mode)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    meth = "evaluate" if args.eval else "run"
    run = get_agent_meth(meth, args.agent)
    run(args, env, device=device, is_atari=is_atari)
