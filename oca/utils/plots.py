"""
oca/utils/plots.py \n
Plotting utilities
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np

from .config import ConfigPlotsProto


def parse_log_file(log_path: str):
    episode_steps = []
    with open(log_path, "r") as file:
        for line in file:
            match = re.search(
                r"ep (\d+) done\. total_steps=(\d+) \| reward=([-\d.]+) \| episode_steps=(\d+)",
                line,
            )
            if match:
                ep = int(match.group(1))
                ep_steps = int(match.group(4))
                episode_steps.append((ep, ep_steps))
    return episode_steps


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def plot_steps_vs_episodes(args: ConfigPlotsProto):
    log_path = os.path.join(args.logdir, args.run_name, "logger.log")
    data = parse_log_file(log_path)
    if not data:
        print("No data found in log.")
        return

    episodes, ep_steps = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ep_steps, alpha=0.2)
    plt.plot(episodes, smooth(ep_steps, args.smooth_window))
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.grid(True, linestyle=":")

    if args.save_path:
        plt.savefig(args.save_path)
        print(f"Plot saved to {args.save_path}")
    else:
        plt.show()


def main(args: ConfigPlotsProto):
    plot_steps_vs_episodes(args)
