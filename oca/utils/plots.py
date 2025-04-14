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
    groups = []
    with open(log_path, "r") as file:
        for line in file:
            match = re.search(
                r"ep (\d+) done\. total_steps=(\d+) \| reward=([-\d.]+) \| episode_steps=(\d+)",
                line,
            )
            if match:
                ep = int(match.group(1))
                total_steps = int(match.group(2))
                reward = float(match.group(3))
                episode_steps = int(match.group(4))
                groups.append((ep, total_steps, reward, episode_steps))
                
    return groups


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def plot_steps_vs_episodes(args: ConfigPlotsProto):
    log_path = os.path.join(args.logdir, args.run_name, "logger.log")
    data = parse_log_file(log_path)
    if not data:
        print("No data found in log.")
        return

    episodes, total_steps, reward, ep_steps = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ep_steps, alpha=0.2)
    plt.plot(episodes, smooth(ep_steps, args.smooth_window))
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    if args.save or args.save_path:
        save_path = args.save_path or os.path.join(
            args.logdir, args.run_name, "steps_vs_episodes.png"
        )
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
        
def plot_rewards_vs_episodes(args: ConfigPlotsProto):
    log_path = os.path.join(args.logdir, args.run_name, "logger.log")
    data = parse_log_file(log_path)
    if not data:
        print("No data found in log.")
        return

    episodes, total_steps, reward, ep_steps = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, reward, alpha=0.2)
    plt.plot(episodes, smooth(reward, args.smooth_window))
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    if args.save or args.save_path:
        save_path = args.save_path or os.path.join(
            args.logdir, args.run_name, "reward_vs_episodes.png"
        )
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main(args: ConfigPlotsProto):
    plot_steps_vs_episodes(args)
    plot_rewards_vs_episodes(args)
